import math
import random

import torch
from torch import nn

from generators.base_function import PixelNorm, EqualLinear, StyledConv, ToRGB, ConvLayer, ResBlock

class Encoder(nn.Module):
    def __init__(
        self,
        size,
        image_style_dim,
        input_dim,
        channel_multiplier=2,
        dropout_rate=0.5,
    ):
        super().__init__()

        self.log_size = int(math.log(size, 2))
        self.size = size

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input_layer = ConvLayer(
            in_channel=input_dim, out_channel=self.channels[size], kernel_size=1
        )
        self.convs = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            in_channels = self.channels[2**i]
            out_channels = self.channels[2**(i-1)]

            conv = nn.Module()
            conv.conv0 = ConvLayer(in_channels, in_channels, kernel_size=3)
            conv.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, downsample=True)

            self.convs.append(conv)
        
        # 4x4
        self.output_layer = ConvLayer(self.channels[4], self.channels[4], kernel_size=3)
        self.output_linear = nn.Sequential(
            EqualLinear(self.channels[4]*4*4, image_style_dim, activation='fused_lrelu'),
            nn.Dropout(dropout_rate)
        ) 
    
    def forward(self, input_image, face_3dmm, mask, ):
        inputs = torch.cat([mask - 0.5, input_image * mask, face_3dmm], 1)
        out = self.input_layer(inputs) 
        encoder_fea = {}
        for index, res in enumerate(range(self.log_size, 2, -1)):
            conv = self.convs[index]
            out = conv.conv0(out)
            encoder_fea[res] = out
            out = conv.conv1(out)

        out = self.output_layer(out)
        encoder_fea[2] = out
        batch_size = out.shape[0]
        style_im = self.output_linear(out.view(batch_size, -1))
        return encoder_fea, style_im


class Decoder(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        image_style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        style_dim = style_dim + image_style_dim 

        self.input = EqualLinear(image_style_dim, self.channels[4]*4*4, activation='fused_lrelu')
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        style_im,
        encoder_fea,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        styles_im = style_im.unsqueeze(1).repeat(1, self.n_latent, 1)
        latent = torch.cat([latent, styles_im], 2)

        
        out = self.input(style_im).view(-1, self.channels[4], 4, 4)
        out = out + encoder_fea[2]
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = out + encoder_fea[i//2 + 3]
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None

class Generator(nn.Module):
    def __init__(
        self,
        size,
        input_dim,
        style_dim,
        image_style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,    
        dropout_rate=0.5,
    ):
        super().__init__()
    
        self.encoder = Encoder(
            size, 
            image_style_dim, 
            input_dim, 
            channel_multiplier, 
            dropout_rate
        )
        self.decoder = Decoder(
            size, 
            style_dim,
            image_style_dim,
            n_mlp,
            channel_multiplier,
            blur_kernel,
            lr_mlp,
        )

    def forward(self, styles, input_image, face_3dmm, mask, return_latents=False):
        encoder_fea, style_im = self.encoder(input_image, face_3dmm, mask)
        image, latent = self.decoder(styles, style_im, encoder_fea, return_latents=return_latents)
        image = input_image * mask + image*(1-mask)
        return image, latent


class Discriminator(nn.Module):
    def __init__(self, size, input_dim, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(input_dim, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

