import math
import numpy as np

import torch
import torch.nn.functional as F
import torchvision

from trainers.base import BaseTrainer
from util.trainer import accumulate
from util.misc import to_cuda
from util.parametric_face_model import ParametricFaceModel
from util.face_renderer import MeshRenderer
from util.misc import draw_landmarks
from loss.kl import GaussianKLLoss
from loss.coeff_loss import CoeffRegLoss, LandmarkLoss, ReflectanceLoss, PhotoLoss, FaceIDLoss, GammaLoss

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

        self.mean_path_length = 0
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

        vis_dataloader = val_data_loader if val_data_loader is not None else train_data_loader
        self.grid_size, self.samples = \
            self.setup_snapshot_image_grid(vis_dataloader)

    def _init_loss(self, opt):
        self._assign_criteria(
            'coeff_regu',
            CoeffRegLoss(opt.trainer.loss_weight).to('cuda'),
            opt.trainer.loss_weight.weight_reg)

        self._assign_criteria(
            'gamma',
            GammaLoss().to('cuda'),
            opt.trainer.loss_weight.weight_gamma)            

        self._assign_criteria(
            'reflect',
            ReflectanceLoss().to('cuda'),
            opt.trainer.loss_weight.weight_reflect)

        self._assign_criteria(
            'lm',
            LandmarkLoss(),
            opt.trainer.loss_weight.weight_lm)

        self._assign_criteria(
            'face_id',
            FaceIDLoss(**opt.trainer.face_id_param),
            opt.trainer.loss_weight.weight_face_id)     
        
        self._assign_criteria(
            'recon',
            PhotoLoss(),
            opt.trainer.loss_weight.weight_recon)                     

        self._assign_criteria(
            'gaussiankl',
            GaussianKLLoss().to('cuda'),
            opt.trainer.loss_weight.weight_kl)

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight

    def optimize_parameters(self, data):
        self.gen_losses = {}
        pred_coeff, mu, logvar = self.net_G(data['input_image'], data['mask'])
        pred_dict = self._split_coeff(pred_coeff)
        pred_face, pred_mask_face, pred_lm, pred_tex \
            = self.render_coeff(pred_coeff)
        pred_mask_face = pred_mask_face.detach()

        self.gen_losses["recon"] = self.criteria['recon'](
            pred_face, data['gt_face'], data['gt_mask_face']*pred_mask_face
        )
        self.gen_losses['face_id'] = self.criteria['face_id'](
            pred_face, data['gt_face'], pred_lm, data['gt_lm']
        )
        self.gen_losses["lm"] = self.criteria['lm'](pred_lm, data['gt_lm']) 
        self.gen_losses['reflect'] = self.criteria['reflect'](
            pred_tex, self.parametric_face_model.skin_mask
        )
        self.gen_losses['coeff_regu'] = self.criteria['coeff_regu'](pred_dict)        
        self.gen_losses['gamma'] = self.criteria['gamma'](pred_dict)        

        if logvar is not None and mu is not None:
            self.gen_losses['gaussiankl'] = \
                self.criteria['gaussiankl'](mu, logvar)

        total_loss = 0 
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]

        self.gen_losses['total_loss'] = total_loss

        self.net_G.zero_grad()
        total_loss.backward()
        self.opt_G.step()

        accumulate(self.net_G_ema, self.net_G_module, self.accum)

    def render_coeff(self, coeff_3dmm, crop_affine=None):
        vertex, tex, color, lm = \
            self.parametric_face_model.compute_for_render(coeff_3dmm)
        mask_face, face = self.mesh_renderer(
            vertex, self.parametric_face_model.face_buf, color)
        if crop_affine is not None:
            grid = F.affine_grid(crop_affine[:,:2,], face.shape, align_corners=False)
            face = F.grid_sample(face, grid, align_corners=False, padding_mode="border")
            mask_face = F.grid_sample(mask_face, grid, align_corners=False, padding_mode="border")

            crop_affine_inv = torch.inverse(crop_affine)
            crop_affine_inv = crop_affine_inv.permute(0,2,1)
            lm[..., 1] = face.shape[-1] - 1 - lm[..., 1]
            lm = lm / face.shape[-1] * 2 - 1
            lm = lm @ crop_affine_inv[:,:2,:2] 
            lm = lm + crop_affine_inv[:,2:,:2]
            lm = (lm + 1)/2.0*face.shape[-1]
            lm[..., 1] = face.shape[-1] - 1 - lm[..., 1]

        return face, mask_face, lm, tex


    def _start_of_iteration(self, data, current_iteration):
        self.batch_size = data['input_image'].shape[0]
        data['input_image'] = (data['input_image']+1)/2.0
        data.update(self._split_coeff(data['coeff_3dmm']))
        with torch.no_grad():
            gt_face, gt_mask_face, gt_lm, gt_tex \
                = self.render_coeff(data['coeff_3dmm'], data['crop_affine'])
            data['gt_face'] = gt_face.detach()
            data['gt_mask_face'] = gt_mask_face.detach()
            data['gt_lm'] = gt_lm.detach()        
        return data
    
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

    def _get_visualizations(self, data):
        with torch.no_grad():
            self.net_G_ema.eval()
            pred_coeff_recon, _, _ = self.net_G_ema(
                self.samples['input_image'],
                self.samples['mask'],
                )
            pred_face_recon, _, pred_lm_recon, _ \
                = self.render_coeff(pred_coeff_recon)                
            pred_face_recon = draw_landmarks(pred_face_recon, pred_lm_recon)

            pred_coeff_random, _, _ = self.net_G_ema(
                self.samples['input_image'],
                self.samples['mask'],
                inference=True
                )
            pred_face_random, _, pred_lm_random, _ \
                = self.render_coeff(pred_coeff_random)                
            pred_face_random = draw_landmarks(pred_face_random, pred_lm_random)

            visualization = torch.cat(
                [self.samples['masked_input'],
                self.samples['input_image'],
                self.samples['gt_face'],
                pred_face_recon,
                pred_face_random], 3
            )
        visualization = torchvision.utils.make_grid(
            visualization, self.grid_size[0])[None]
        visualization = visualization * 2 - 1
        return visualization

    def setup_snapshot_image_grid(self, data_set, size='1080p',):
        # Select size.
        gw = 1; gh = 1
        if size == '1080p':
            gw = np.clip(1920 // data_set.dataset.resolution, 3, 32)
            gh = np.clip(1080 // data_set.dataset.resolution, 2, 32)
        if size == '4k':
            gw = np.clip(3840 // data_set.dataset.resolution, 7, 32)
            gh = np.clip(2160 // data_set.dataset.resolution, 4, 32)
        if size == '8k':
            gw = np.clip(7680 // data_set.dataset.resolution, 7, 32)
            gh = np.clip(4320 // data_set.dataset.resolution, 4, 32)

        # Random layout.
        samples = data_set.dataset.get_minibatch_val_np(gw * gh)
        samples = to_cuda(samples)
        samples['input_image'] = (samples['input_image']+1)/2.0
        samples['gt_face'], samples['gt_face_mask'], lm, _ \
                = self.render_coeff(samples['coeff_3dmm'], samples['crop_affine'])          

        samples['masked_input'] = samples['input_image'] * samples['mask']
        samples['gt_face'] = draw_landmarks(samples['gt_face'], lm)
        return (gw, gh), samples

