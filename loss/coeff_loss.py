import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from kornia.geometry import warp_affine
from skimage import transform as trans
from PIL import Image

from third_part.arcface_torch.backbones import get_model as arcface_get_model

class CoeffRegLoss(nn.Module):
    def __init__(self, weights) -> None:
        super().__init__()
        self.w_id = weights.weight_id
        self.w_exp = weights.weight_exp
        self.w_tex = weights.weight_tex

    def forward(self, coeffs_dict):
        """
        l2 norm without the sqrt, from yu's implementation (mse)
        tf.nn.l2_loss https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
        Parameters:
            coeffs_dict     -- a  dict of torch.tensors , keys: id, exp, tex, angle, gamma, trans

        """
        # coefficient regularization to ensure plausible 3d faces
        creg_loss = self.w_id * torch.sum(coeffs_dict['id'] ** 2) +  \
            self.w_exp * torch.sum(coeffs_dict['exp'] ** 2) + \
            self.w_tex * torch.sum(coeffs_dict['tex'] ** 2)
        creg_loss = creg_loss / coeffs_dict['id'].shape[0]

        return creg_loss


class GammaLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, coeffs_dict):
        # gamma regularization to ensure a nearly-monochromatic light
        gamma = coeffs_dict['gamma'].reshape([-1, 3, 9])
        gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
        gamma_loss = torch.mean((gamma - gamma_mean) ** 2)

        return gamma_loss


class ReflectanceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, texture, mask):
        """
        minimize texture variance (mse), albedo regularization to ensure an uniform skin albedo
        Parameters:
            texture       --torch.tensor, (B, N, 3)
            mask          --torch.tensor, (N), 1 or 0

        """        
        mask = mask.reshape([1, mask.shape[0], 1])
        texture_mean = torch.sum(mask * texture, dim=1, keepdims=True) / torch.sum(mask)
        loss = torch.sum(((texture - texture_mean) * mask)**2) / (texture.shape[0] * torch.sum(mask))
        return loss

class PhotoLoss(nn.Module):        
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, imageA, imageB, mask, eps=1e-6):
        """
        l2 norm (with sqrt, to ensure backward stabililty, use eps, otherwise Nan may occur)
        Parameters:
            imageA       --torch.tensor (B, 3, H, W), range (0, 1), RGB order 
            imageB       --same as imageA
        """
        loss = torch.sqrt(eps + torch.sum((imageA - imageB) ** 2, dim=1, keepdims=True)) * mask
        loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
        return loss


class LandmarkLoss(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is None:
            weight = np.ones([68])
            weight[28:31] = 20
            weight[-8:] = 20
            weight = np.expand_dims(weight, 0)
            weight = torch.tensor(weight).to('cuda')            
        self.weight = weight
        
    def forward(self, inp, target):
        loss = torch.sum((inp - target)**2, dim=-1) * self.weight
        loss = torch.sum(loss) / (inp.shape[0] * inp.shape[1])
        return loss

class FaceIDLoss(nn.Module):
    def __init__(self, network, pretrained_path, image_size, perceptual_input_size ):
        super().__init__()
        self.model = arcface_get_model(name=network, fp16=False)
        state_dict = torch.load(pretrained_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.preprocess = lambda x: 2 * x - 1
        self.image_size=image_size
        self.perceptual_input_size=perceptual_input_size

    def forward(self, inp, target, inp_landmark, target_landmark):
        M = self.estimate_norm_torch(inp_landmark, self.image_size)
        inp = self.preprocess(self.resize_n_crop(inp, M))
        # Image.fromarray(((inp.clone().detach().cpu().permute(0, 2, 3, 1).numpy()[0]+1)/2*255).astype(np.uint8)).save('test_inp.png')
        id_input = F.normalize(self.model(inp), dim=-1, p=2)

        M = self.estimate_norm_torch(target_landmark, self.image_size)
        target = self.preprocess(self.resize_n_crop(target, M))
        # Image.fromarray(((target.clone().detach().cpu().permute(0, 2, 3, 1).numpy()[0]+1)/2*255).astype(np.uint8)).save('target_inp.png')
        # exit()
        id_target = F.normalize(self.model(target), dim=-1, p=2)

        cosine_d = torch.sum(id_input * id_target, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
        return torch.sum(1 - cosine_d) / cosine_d.shape[0] 

    def resize_n_crop(self, image, M):
        # image: (b, c, h, w)
        # M   :  (b, 2, 3)
        return warp_affine(
            image, M, dsize=(self.perceptual_input_size, self.perceptual_input_size)
        )

    # utils for face recognition model
    def estimate_norm(self, lm_68p, H):
        # from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
        """
        Return:
            trans_m            --numpy.array  (2, 3)
        Parameters:
            lm                 --numpy.array  (68, 2), y direction is opposite to v direction
            H                  --int/float , image height
        """
        lm = self.extract_5p(lm_68p)
        lm[:, -1] = H - 1 - lm[:, -1]
        tform = trans.SimilarityTransform()
        assert self.perceptual_input_size == 112
        src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)

        tform.estimate(lm, src)
        M = tform.params
        if np.linalg.det(M) == 0:
            M = np.eye(3)

        return M[0:2, :]

    def estimate_norm_torch(self, lm_68p, H):
        lm_68p_ = lm_68p.detach().cpu().numpy()
        M = []
        for i in range(lm_68p_.shape[0]):
            M.append(self.estimate_norm(lm_68p_[i], H))
        M = torch.tensor(np.array(M), dtype=torch.float32).to(lm_68p.device)
        return M

    def extract_5p(self, lm):
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
            lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
        lm5p = lm5p[[1, 2, 0, 3, 4], :]
        return lm5p


class Landmark5PointLoss(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is None:
            weight = np.ones([1, 5])
            # weight[28:31] = 20
            # weight[-8:] = 20
            # weight = np.expand_dims(weight, 0)
            weight = torch.tensor(weight).to('cuda')            
        self.weight = weight
        self.lm_idx = torch.tensor([31, 37, 40, 43, 46, 49, 55]) - 1
        
    def forward(self, inp, target):
        if inp.shape[1] == 68:
            inp = self.extract_5p_torch(inp)
        if target.shape[1] == 68:
            target = self.extract_5p_torch(target)

        loss = torch.sum((inp - target)**2, dim=-1) * self.weight
        loss = torch.sum(loss) / (inp.shape[0] * inp.shape[1])
        return loss

    def extract_5p_torch(self, lm):
        # lm = lm.clone()
        lm_eye_left = torch.mean(
            torch.cat([lm[:,self.lm_idx[1],][:,None], 
                       lm[:,self.lm_idx[2],][:,None]], 1), 1, keepdim=True)
        
        lm_eye_right = torch.mean(
            torch.cat([lm[:,self.lm_idx[3],][:,None], 
                       lm[:,self.lm_idx[4],][:,None]], 1), 1, keepdim=True)
                       
        lm5p = torch.cat([lm[:, self.lm_idx[0],][:,None],
                          lm_eye_left,
                          lm_eye_right,
                          lm[:, self.lm_idx[5],][:,None],
                          lm[:, self.lm_idx[6],][:,None],], 1)
        return lm5p        