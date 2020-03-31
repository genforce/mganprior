import torch
import torch.nn as nn
import torch.nn.functional as F

from .gan_utils import get_gan_model


PGGAN_LATENT_1024 = [(512, 1, 1),
              (512, 4, 4), (512, 4, 4),
              (512, 8, 8), (512, 8, 8),
              (512, 16, 16), (512, 16, 16),
              (512, 32, 32), (512, 32, 32),
              (256, 64, 64), (256, 64, 64),
              (128, 128, 128), (128, 128, 128),
              (64, 256, 256), (64, 256, 256),
              (32, 512, 512), (32, 512, 512),
              (16, 1024, 1024), (16, 1024, 1024),
              (3, 1024, 1024)]

PGGAN_LATENT_256 = [(512, 1, 1), (512, 4, 4),
              (512, 4, 4), (512, 8, 8),
              (512, 8, 8), (512, 16, 16),
              (512, 16, 16), (512, 32, 32),
              (512, 32, 32), (256, 64, 64),
              (256, 64, 64), (128, 128, 128),
              (128, 128, 128), (64, 256, 256),
              (64, 256, 256), (3, 256, 256)]

PGGAN_LAYER_MAPPING = {  # The new PGGAN includes the intermediate output layer, need mapping
    0: 0, 1: 1, 2: 3, 3: 4, 4: 6, 5: 7, 6: 9, 7: 10, 8: 12
}


def get_derivable_generator(gan_model_name, generator_type, args):
    if generator_type == 'PGGAN-z':  # Single latent code
        return PGGAN(gan_model_name)
    elif generator_type == 'StyleGAN-z':
        return StyleGAN(gan_model_name, 'z')
    elif generator_type == 'StyleGAN-w':
        return StyleGAN(gan_model_name, 'w')
    elif generator_type == 'StyleGAN-w+':
        return StyleGAN(gan_model_name, 'w+')
    elif generator_type == 'PGGAN-Multi-Z':  # Multiple Latent Codes
        return PGGAN_multi_z(gan_model_name, args.composing_layer, args.z_number, args)
    else:
        raise Exception('Please indicate valid `generator_type`')


class PGGAN(nn.Module):
    def __init__(self, gan_model_name):
        super(PGGAN, self).__init__()
        self.pggan = get_gan_model(gan_model_name)
        self.init = False

    def input_size(self):
        return [(512,)]

    def cuda(self, device=None):
        self.pggan.cuda(device=device)

    def forward(self, z):
        latent = z[0]
        return self.pggan(latent)


class StyleGAN(nn.Module):
    def __init__(self, gan_model_name, start):
        super(StyleGAN, self).__init__()
        self.stylegan = get_gan_model(gan_model_name).net
        self.start = start
        self.init = False

    def input_size(self):
        if self.start == 'z' or self.start == 'w':
            return [(512,)]
        elif self.start == 'w+':
            return [(self.stylegan.net.synthesis.num_layers, 512)]

    def cuda(self, device=None):
        self.stylegan.cuda(device=device)

    def forward(self, latent):
        z = latent[0]
        if self.start == 'z':
            w = self.stylegan.net.mapping(z)
            w = self.stylegan.net.truncation(w)
            x = self.stylegan.net.synthesis(w)
            return x
        elif self.start == 'w':
            z = z.view((-1, self.stylegan.net.synthesis.num_layers, self.stylegan.net.synthesis.w_space_dim))
            x = self.stylegan.net.synthesis(z)
            return x
        elif self.start == 'w+':
            x = self.stylegan.net.synthesis(z)
            return x


class PGGAN_multi_z(nn.Module):
    def __init__(self, gan_model_name, blending_layer, z_number, args):
        super(PGGAN_multi_z, self).__init__()
        self.blending_layer = blending_layer
        self.z_number = z_number
        self.z_dim = 512
        self.pggan = get_gan_model(gan_model_name)

        self.pre_model = nn.Sequential(*list(self.pggan.children())[:blending_layer])
        self.post_model = nn.Sequential(*list(self.pggan.children())[blending_layer:])
        self.init = True


        PGGAN_LATENT = PGGAN_LATENT_1024 if gan_model_name == 'PGGAN-CelebA' else PGGAN_LATENT_256
        self.mask_size = PGGAN_LATENT[blending_layer][1:]
        self.layer_c_number = PGGAN_LATENT[blending_layer][0]

    def input_size(self):
        return [(self.z_number, self.z_dim), (self.z_number, self.layer_c_number)]

    def init_value(self, batch_size):
        z_estimate = torch.randn((batch_size, self.z_number, self.z_dim)).cuda()  # our estimate, initialized randomly
        z_alpha = torch.full((batch_size, self.z_number, self.layer_c_number), 1 / self.z_number).cuda()
        return [z_estimate, z_alpha]

    def cuda(self, device=None):
        self.pggan.cuda(device=device)

    def forward(self, z):
        z_estimate, alpha_estimate = z
        feature_maps_list = []
        for j in range(self.z_number):
            feature_maps_list.append(
                self.pre_model(z_estimate[:, j, :].view((-1, self.z_dim, 1, 1))) * alpha_estimate[:, j, :].view((-1, self.layer_c_number, 1, 1)))
        fused_feature_map = sum(feature_maps_list) / self.z_number
        y_estimate = self.post_model(fused_feature_map)
        return y_estimate

