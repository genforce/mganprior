import numpy as np
import os

import torch
import torch.nn as nn
from models.helper import build_generator


PGGAN_Inter_Output_Layer_256 = [-1, 17, 14, 11, 8, 5, 2]
PGGAN_Inter_Output_Layer_1024 = [-1, 23, 20, 17, 14, 11, 8, 5, 2]


def standard_z_sample(size, depth, device=None):
    '''
    Generate a standard set of random Z as a (size, z_dimension) tensor.
    With the same random seed, it always returns the same z (e.g.,
    the first one is always the same regardless of the size.)
    '''
    # Use numpy RandomState since it can be done deterministically
    # without affecting global state
    rng = np.random.RandomState(None)
    result = torch.from_numpy(rng.standard_normal(size * depth).reshape(size, depth)).float()
    if device is not None:
        result = result.to(device)
    return result


def get_gan_model(model_name):
    """
    :param model_name: Please refer `GAN_MODELS`
    :return: gan_model(nn.Module or nn.Sequential)
    """
    gan = build_generator(model_name)
    if model_name.startswith('pggan'):
        gan_list = list(gan.net.children())
        remove_index = PGGAN_Inter_Output_Layer_1024 if model_name == 'pggan_celebahq' else PGGAN_Inter_Output_Layer_256
        for output_index in remove_index:
            gan_list.pop(output_index)
        return nn.Sequential(*gan_list)
    elif model_name.startswith('style'):
        return gan




