import os
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

from config import Config
from util.distributed import init_dist
from util.logging import init_logging, make_logging_dir
from util.io import tensor_to_pilimage, save_pilimage_in_png
from util.misc import to_cuda, make_noise
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/facial_image_renderer_ffhq.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--mask_dir', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)

    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = torch.cuda.current_device()
    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)

    # create a model
    net_G, net_D, net_G_ema, opt_G, opt_D, sch_G, sch_D \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_D, net_G_ema, \
                          opt_G, opt_D, sch_G, sch_D, \
                          train_dataset=None)

    current_epoch, current_iteration = trainer.load_checkpoint(opt, args.which_iter)                          

    output_dir = os.path.join(
        args.output_dir, 
        'epoch_{:05}_iteration_{:09}'.format(current_epoch, current_iteration)
        )
    os.makedirs(output_dir, exist_ok=True)
    image_list = sorted(glob.glob(os.path.join(args.input_dir,'*.png')))
    mask_list = sorted(glob.glob(os.path.join(args.mask_dir,'*.png')))

    img_trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])    

    net_G = trainer.net_G_ema.eval()
    with torch.no_grad():
        for image_file, mask_file in tqdm(zip(image_list, mask_list)):
            im = Image.open(image_file).convert('RGB')
            gt = img_trans(im).to('cuda')[None]
            mask = (np.array(Image.open(mask_file))>128).astype(np.float32)
            mask = torch.tensor(mask).to('cuda').permute(2,0,1)[None]
            mask = mask[:,:1,:,:]

            input_image = gt*mask

            coeff_noise = make_noise(input_image.shape[0], 
                                    opt.semantic_recommender.param.style_dim,
                                    args.num_samples, opt.device)        
            coeff_input = (input_image+1)/2.
            pred_coeffs = trainer.semantic_recom.inference(coeff_input,mask,coeff_noise)

            fake_imgs, pred_faces = [], []
            for i in range(args.num_samples):
                noise = [make_noise(input_image.shape[0],
                                    opt.trainer.latent, 1, 'cuda')]
                pred_coeff = pred_coeffs[i]
                pred_face, _ = trainer.render_coeff(pred_coeff)   

                fake_img, _ = net_G(noise, input_image, pred_face, mask)   
                fake_img = fake_img.detach().cpu()
                fake_imgs.append(fake_img)
                pred_face = pred_face.detach().cpu() 
                pred_faces.append(pred_face)

            image_name = os.path.basename(image_file)
            img1 = torch.cat([input_image.detach().cpu()]+fake_imgs,3)
            img0 = torch.cat([gt.detach().cpu()]+pred_faces, 3)
            img = torch.cat([img0, img1], 2)
            img_all = tensor_to_pilimage(img)

            name = os.path.join(output_dir, image_name.replace('.png', f'_all.jpg'))
            save_pilimage_in_png(name, img_all[0])
    print('done')