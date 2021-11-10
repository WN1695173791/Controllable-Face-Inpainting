import os
import math
import random
import importlib
import numpy as np
from PIL import Image

import torch
from torch import autograd
import torch.nn.functional as F
import torchvision

from trainers.base import BaseTrainer
from util.io import tensor_to_pilimage, save_pilimage_in_png
from util.trainer import accumulate
from util.misc import to_cuda, make_noise
from util.parametric_face_model import ParametricFaceModel
from util.face_renderer import MeshRenderer
from loss.gan  import GANLoss

class Trainer(BaseTrainer):
    def __init__(self, opt, net_G, net_D, net_G_ema, 
                 opt_G, opt_D, sch_G, sch_D, 
                 train_data_loader, val_data_loader=None):                 
        super(Trainer, self).__init__(
            opt, net_G, net_D, net_G_ema, 
            opt_G, opt_D, sch_G, sch_D, 
            train_data_loader, val_data_loader)

        self.accum = 0.5 ** (32 / (10 * 1000))
        self.log_size = int(math.log(opt.data.resolution, 2))

        self.parametric_face_model = ParametricFaceModel(
            bfm_folder=opt.camera.bfm_folder, camera_distance=opt.camera.camera_d, 
            focal=opt.camera.focal, center=opt.camera.center,
            default_name=opt.camera.bfm_model
        ).to('cuda')

        fov = 2 * np.arctan(opt.camera.center / opt.camera.focal) * 180 / np.pi
        self.mesh_renderer = MeshRenderer(
            rasterize_fov=fov,
            znear=opt.camera.z_near, 
            zfar=opt.camera.z_far, 
            rasterize_size=int(2 * opt.camera.center)
            )

        self.semantic_recom = self.load_semantic_recom().eval()
        if not self.is_inference:
            data_loader = val_data_loader if val_data_loader is not None else train_data_loader
            self.grid_size, self.samples = \
                self.setup_snapshot_image_grid(data_loader)
                
    def _init_loss(self, opt):
        self._assign_criteria(
            'gan',
            GANLoss(gan_mode=opt.trainer.gan_mode).to('cuda'),
            opt.trainer.loss_weight.weight_gan)

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight

    def optimize_parameters(self, data):
        self.gen_losses = {}
        noise = self.mixing_noise(
            self.batch_size, 
            self.opt.trainer.latent, 
            self.opt.trainer.mixing, 
            'cuda'
        )
        fake_img, _ = self.net_G(
            noise, 
            data['input_image'], 
            data['gt_face'], 
            data['mask']
        )

        fake_pred = self.net_D(torch.cat([fake_img, data['gt_face']], 1))
        g_loss = self.criteria['gan'](fake_pred, t_real=True, dis_update=False)
        self.gen_losses["gan"] = g_loss 

        self.net_G.zero_grad()
        g_loss.backward()
        self.opt_G.step()

        accumulate(self.net_G_ema, self.net_G_module, self.accum)

        self.dis_losses = {}
        fake_pred = self.net_D(torch.cat([fake_img, data['gt_face']], 1).detach())
        real_pred = self.net_D(torch.cat([data['input_image'], data['gt_face']], 1))
        fake_loss = self.criteria['gan'](fake_pred, t_real=False, dis_update=True)
        real_loss = self.criteria['gan'](real_pred, t_real=True,  dis_update=True)
        d_loss = fake_loss + real_loss
        self.dis_losses["d"] = d_loss
        self.dis_losses["real_score"] = real_pred.mean()
        self.dis_losses["fake_score"] = fake_pred.mean()        

        self.net_D.zero_grad()
        d_loss.backward()
        self.opt_D.step()

        if self.d_regularize:
            inputs = torch.cat([data['input_image'], data['gt_face']], 1)
            inputs.requires_grad = True
            real_pred = self.net_D(inputs)
            r1_loss = self.d_r1_loss(real_pred, inputs)

            self.net_D.zero_grad()
            (self.opt.trainer.r1 / 2 * r1_loss * self.opt.trainer.d_reg_every + 0 * real_pred[0]).backward()
            self.opt_D.step()

            self.dis_losses["r1"] = r1_loss

    def d_r1_loss(self, real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def mixing_noise(self, batch, latent_dim, prob, device):
        if prob > 0 and random.random() < prob:
            return make_noise(batch, latent_dim, 2, device)
        else:
            return [make_noise(batch, latent_dim, 1, device)]

    def render_coeff(self, coeff_3dmm, crop_affine=None):
        vertex, tex, color, lm = \
            self.parametric_face_model.compute_for_render(coeff_3dmm)
        mask_face, face = self.mesh_renderer(
            vertex, self.parametric_face_model.face_buf, color)
        if crop_affine is not None:
            grid = F.affine_grid(crop_affine[:,:2,], face.shape, align_corners=False)
            face = F.grid_sample(face, grid, align_corners=False, padding_mode="border")
            mask_face = F.grid_sample(mask_face, grid, align_corners=False, padding_mode="border")
        face = face*2.-1.
        return face, mask_face

    def _start_of_iteration(self, data, current_iteration):
        self.d_regularize = current_iteration % self.opt.trainer.d_reg_every == 0
        self.batch_size = data['input_image'].shape[0]
        data.update(self._split_coeff(data['coeff_3dmm']))
        with torch.no_grad():
            gt_face, gt_mask_face \
                = self.render_coeff(data['coeff_3dmm'], data['crop_affine'])
            data['gt_face'] = gt_face.detach()
            data['gt_mask_face'] = gt_mask_face.detach()
        return data

    def load_semantic_recom(self):
        gen_module, gen_network_name = self.opt.semantic_recommender.type.split('::')
        lib = importlib.import_module(gen_module)
        network = getattr(lib, gen_network_name)
        net = network(**self.opt.semantic_recommender.param).to(self.opt.device)
        checkpoint = torch.load(
            self.opt.semantic_recommender.load_path, 
            map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['net_G_ema'])
        return net

    def _get_visualizations(self, data):
        with torch.no_grad():
            self.net_G_ema.eval()
            fake_image_gt, _ = self.net_G_ema(
                self.samples['styles'],
                self.samples['input_image'],
                self.samples['gt_face'],
                self.samples['mask'],
                )

            coeffs = self.semantic_recom.inference(
                (self.samples['input_image']+1)/2.,
                self.samples['mask'],
                make_noise(
                    self.samples['input_image'].shape[0], 
                    self.opt.semantic_recommender.param.style_dim,
                    5, self.opt.device)
            )    
            fake_images = []            
            face_coeffs = []            
            for coeff in coeffs:
                face_coeff,_ = self.render_coeff(coeff)
                fake_image, _ = self.net_G_ema(
                    self.samples['styles'],
                    self.samples['input_image'],
                    face_coeff,
                    self.samples['mask'],
                    )
                fake_images.append(fake_image.detach().cpu())
                face_coeffs.append(face_coeff.detach().cpu())

            visualization_down = torch.cat(
                [self.samples['masked_input'].detach().cpu(),
                fake_image_gt.detach().cpu()]+fake_images,  3
            )
            visualization_up = torch.cat(
                [self.samples['masked_input'].detach().cpu(),
                self.samples['masked_face'].detach().cpu(),]+face_coeffs,  3
            )
            visualization = torch.cat([
                visualization_up, 
                visualization_down], 2).clamp_(-1, 1)
        visualization = torchvision.utils.make_grid(
            visualization, self.grid_size[0])[None]
        return visualization
    
    def setup_snapshot_image_grid(
        self, train_set,
        size='720p',     # '1080p' = to be viewed on 1080p display, '4k' = to be viewed on 4k display.
        ):
        # Select size.
        gw = 1; gh = 1
        if size == '720p':
            gw = np.clip(1280 // train_set.dataset.resolution, 3, 32)
            gh = np.clip(720 // train_set.dataset.resolution, 3, 32)
        if size == '1080p':
            gw = np.clip(1920 // train_set.dataset.resolution, 3, 32)
            gh = np.clip(1080 // train_set.dataset.resolution, 2, 32)
        if size == '4k':
            gw = np.clip(3840 // train_set.dataset.resolution, 7, 32)
            gh = np.clip(2160 // train_set.dataset.resolution, 4, 32)
        if size == '8k':
            gw = np.clip(7680 // train_set.dataset.resolution, 7, 32)
            gh = np.clip(4320 // train_set.dataset.resolution, 4, 32)
        
        # Random layout.
        samples = train_set.dataset.get_minibatch_val_np(gw * gh)
        samples = to_cuda(samples)
        samples['styles'] = [to_cuda(torch.randn(gw*gh, self.opt.trainer.latent))]
        samples['gt_face'], samples['gt_mask_face'] = \
            self.render_coeff(samples['coeff_3dmm'], samples['crop_affine'])
        samples['masked_input'] = samples['input_image'] * samples['mask']
        samples['masked_face'] = \
            samples['input_image'] * (1-samples['gt_mask_face']) \
            + samples['gt_face'] * samples['gt_mask_face']

        return (gw, gh), samples

    def _split_coeff(self, coeff):
        id_coeff = coeff[:, :80]
        exp_coeff = coeff[:, 80: 144]
        tex_coeff = coeff[:, 144: 224]
        angles = coeff[:, 224: 227]
        gammas = coeff[:, 227: 254]
        translations = coeff[:, 254:]
        return {
            'id': id_coeff,
            'exp': exp_coeff,
            'tex': tex_coeff,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }


    def test(self, data, output_dir, current_iteration, num_samples=5):
        r"""Compute results images for a batch of input data and save the
        results in the specified folder.

        Args:
            data (dict): a batch of data.
            output_dir (str): Target location for saving the output image.
        """
        self.net_G_ema.eval()
        with torch.no_grad():
            coeff_noise = make_noise(data['input_image'].shape[0], 
                                    self.opt.semantic_recommender.param.style_dim,
                                    num_samples, self.opt.device)        
            coeff_input = (data['input_image']+1)/2.
            pred_coeffs = self.semantic_recom.inference(coeff_input,data['mask'],coeff_noise)


            fake_imgs, pred_faces = [], []
            for i in range(num_samples):
                noise = [make_noise(self.batch_size, 
                                    self.opt.trainer.latent, 1, 'cuda')]
                pred_coeff = pred_coeffs[i]
                pred_face, _ = self.render_coeff(pred_coeff)      


                fake_img, _ = self.net_G_ema(noise, data['input_image'], pred_face, data['mask'])   
                fake_img = fake_img.detach().cpu()
                fake_imgs.append(fake_img)
                pred_face = pred_face.detach().cpu() 
                pred_faces.append(pred_face)

                fake_img_pil = tensor_to_pilimage(fake_img)
                pred_face_pil = tensor_to_pilimage(pred_face)
                for batch, (img, face) in enumerate(zip(fake_img_pil, pred_face_pil)):
                    image_name = data['key'][batch]
                    name = os.path.join(output_dir, image_name.replace('.png', f'_gen{i}.png'))
                    save_pilimage_in_png(name, img)
                    name = os.path.join(output_dir, image_name.replace('.png', f'_face{i}.jpg'))
                    save_pilimage_in_png(name, face)     
                
            gt_image_pil = tensor_to_pilimage(data['input_image'])
            input_image_pil = tensor_to_pilimage(data['input_image']*data['mask'])
            mask_pil = tensor_to_pilimage(data['mask'].expand(data['input_image'].shape), norm=False)

            inputs = (data['input_image']*data['mask']).detach().cpu()
            img1 = torch.cat([inputs]+fake_imgs,3)

            img0 = torch.cat([data['input_image'].detach().cpu()]+pred_faces, 3)
            img = torch.cat([img0, img1], 2)
            img_all = tensor_to_pilimage(img)
                        
            for batch, (gt, input, mask, img) in enumerate(zip(gt_image_pil, input_image_pil, mask_pil, img_all)):
                image_name = data['key'][batch]
                name = os.path.join(output_dir, image_name.replace('.png', f'_gt.png'))
                save_pilimage_in_png(name, gt)
                name = os.path.join(output_dir, image_name.replace('.png', f'_input.jpg'))
                save_pilimage_in_png(name, input)
                name = os.path.join(output_dir, image_name.replace('.png', f'_mask.png'))
                save_pilimage_in_png(name, mask)                
                name = os.path.join(output_dir, image_name.replace('.png', f'_all.jpg'))
                save_pilimage_in_png(name, img)

    