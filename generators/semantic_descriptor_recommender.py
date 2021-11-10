import os

import torch
import torch.nn as nn

from generators.resnet import func_dict
from generators.base_function import PixelNorm

class Encoder(nn.Module):
    fc_dim=257
    def __init__(self, net_recon, use_last_fc=False):
        super(Encoder, self).__init__()
        self.use_last_fc = use_last_fc
        if net_recon not in func_dict:
            return  NotImplementedError('network [%s] is not implemented', net_recon)
        func, last_dim = func_dict[net_recon]
        backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)

        self.backbone = backbone
        self.last_dim = last_dim

    def forward(self, x):
        x = self.backbone(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, n_mlp, convert_dim, style_dim, latent_dim):
        super().__init__()
        self.linear_mu = nn.Linear(input_dim, style_dim)
        self.linear_var = nn.Linear(input_dim, style_dim)

        self.input_layer = nn.Sequential(
            PixelNorm(),
            nn.Linear(style_dim, latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(latent_dim)
        )

        self.linear_layers = nn.ModuleList()
        self.convert_layers = nn.ModuleList()
        for i in range(n_mlp):
            self.linear_layers.append(
                nn.Sequential(
                    nn.Linear(latent_dim+convert_dim, latent_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.BatchNorm1d(latent_dim)
                )
            )
            self.convert_layers.append(
                nn.Linear(input_dim, convert_dim),
            )
        
        self.final_layers = nn.ModuleList([
            nn.Linear(latent_dim+input_dim, 80, bias=True),  # id layer
            nn.Linear(latent_dim+input_dim, 64, bias=True),  # exp layer
            nn.Linear(latent_dim+input_dim, 80, bias=True),  # tex layer
            nn.Linear(latent_dim+input_dim, 3,  bias=True),  # angle layer
            nn.Linear(latent_dim+input_dim, 27, bias=True),  # gamma layer
            nn.Linear(latent_dim+input_dim, 2,  bias=True),  # tx, ty
            nn.Linear(latent_dim+input_dim, 1,  bias=True),  # tz
        ])
        for m in self.final_layers:
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)  

        self.style_dim = style_dim 
          
    def forward(self, x, x_reverse=None):
        coeff_list = []
        if x_reverse is not None:
            x_reverse = torch.flatten(x_reverse, 1)
            mu = self.linear_mu(x_reverse)
            logvar = self.linear_var(x_reverse)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            style = eps.mul(std) + mu
        else:
            style = torch.randn(
                x.shape[0], self.style_dim).to('cuda')
            logvar, mu = None, None

        latent = self.input_layer(style)
        x = torch.flatten(x, 1)
        for linear_layer, convert_layer in zip(self.linear_layers, self.convert_layers):
            condition_feature = convert_layer(x)
            latent = linear_layer(torch.cat([latent, condition_feature], 1))

        for index, layer in enumerate(self.final_layers):
            coeff_list.append(layer(torch.cat([latent, x], 1)))
        coeff_3dmm = torch.cat(coeff_list, 1)

        return coeff_3dmm, mu, logvar

    def inference(self, x, noises):
        all_coeff_list = []
        for noise in noises:
            coeff_list = []
            latent = self.input_layer(noise)
            x = torch.flatten(x, 1)
            for linear_layer, convert_layer in zip(self.linear_layers, self.convert_layers):
                condition_feature = convert_layer(x)
                latent = linear_layer(torch.cat([latent, condition_feature], 1))

            for index, layer in enumerate(self.final_layers):
                coeff_list.append(layer(torch.cat([latent, x], 1)))
            coeff_3dmm = torch.cat(coeff_list, 1)
            all_coeff_list.append(coeff_3dmm)

        return all_coeff_list 



class Generator(nn.Module):
    def __init__(
        self,
        model = 'resnet50',
        init_path = None,
        n_mlp=8, 
        convert_dim=512, 
        style_dim=512,
        latent_dim=512,      
    ):
        super().__init__()
        self.encoder = Encoder(model)
        self.decoder = Decoder(self.encoder.last_dim, n_mlp, convert_dim, style_dim, latent_dim)
        if init_path and os.path.isfile(init_path):
            state_dict = self.filter_state_dict(torch.load(init_path, map_location='cpu'))
            self.load_state_dict(state_dict, strict=False)
            print("loading init [image encoder] from %s" %(init_path))
        else:
            print("Do not init [image encoder] !! please check %s" %(init_path))


    def forward(self, input_image, mask, inference=False):
        input_corrupted = input_image*mask
        x = self.encoder(input_corrupted)
        if not inference:
            input_reverse = input_image*(1-mask)
            x_reverse = self.encoder(input_reverse)
        else:
            x_reverse = None 

        coeff_3dmm, mu, logvar = self.decoder(x, x_reverse)

        return coeff_3dmm, mu, logvar

    def inference(self, input_image, mask, noises):
        input_corrupted = input_image*mask
        x = self.encoder(input_corrupted)
        coeff_3dmm = self.decoder.inference(x, noises)
        return coeff_3dmm

    def filter_state_dict(self, state_dict):
        new_state_dict = {}
        for key in state_dict['net_recon']:
            if 'backbone' in key:
                key_new = 'encoder.'+key
                new_state_dict[key_new] = state_dict['net_recon'][key]
        return new_state_dict
