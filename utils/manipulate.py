import numpy as np
import cv2
from math import sqrt, ceil
from PIL import Image

import torch
import torch.nn.functional as F

from .file_utils import Tensor2PIL, PIL2Tensor
from .image_precossing import _add_batch_one

BOUNDARY_DIR = './boundaries'
LEVEL = {  # for style mixing
    'coarse': (0, 4),
    'middle': (4, 8),
    'fine'  : (8, 18)
}
M = torch.Tensor([[0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227]])



def SR_loss(loss_function, down_type='bilinear', factor=8):
    def loss(x, gt):
        x = F.interpolate(x, scale_factor=1/factor, mode=down_type)
        gt = F.interpolate(gt, scale_factor=1/factor, mode=down_type)
        return loss_function(x, gt)
    return loss


def Color_loss(loss_function):
    # Gray = R*0.299 + G*0.587 + B*0.114
    def loss(x, g):
        xt = x[:, 0:1, :, :] * 0.299 + x[:, 1:2, :, :] * 0.587 + x[:, 2:3, :, :] * 0.114
        gt = g[:, 0:1, :, :] * 0.299 + g[:, 1:2, :, :] * 0.587 + g[:, 2:3, :, :] * 0.114
        return loss_function(xt.expand(-1, 3, -1, -1), gt.expand(-1, 3, -1, -1))
    return loss


def get_grid(images, num_rows, res):
    num_images = images.shape[0]
    num_cols = num_images // num_rows + bool(num_images % num_rows)
    result = np.ones((res * num_rows, res * num_cols, 3), np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx == num_images:
                break
            image = cv2.resize(images[idx][:, :, ::-1], (res, res))
            result[res*i:res*(i+1), res*j:res*(j+1), :] = image
    return result


def combine_inpainting(gt, out, mask):
    gt_masked = gt * (1 - mask)
    x_mask = out * mask
    return gt_masked + x_mask


def parsing_mask(mask_path):
    mask = PIL2Tensor(Image.open(mask_path))
    return _add_batch_one(mask)[:, :3, :, :]


def masked_loss(loss_function, mask_tensor):
    def loss(x, gt):
        x_mask = x * (1 - mask_tensor)
        gt_mask = gt * (1 - mask_tensor)
        return loss_function(x_mask, gt_mask)
    return loss


def mask_images(image, mask):
    return image * (1 - mask)


def downsample_images(image, factor, mode):
    down = F.interpolate(image, scale_factor=1/factor, mode=mode)
    up_nn = F.interpolate(down, scale_factor=factor, mode='nearest')
    up_bic = F.interpolate(down, scale_factor=factor, mode='bilinear')
    return up_nn, up_bic


def colorization_images(image):
    return image[:, 0:1, :, :] * 0.299 + image[:, 1:2, :, :] * 0.587 + image[:, 2:3, :, :] * 0.114


def crop_tensor(batch_image_tensor, top, left, bottom, right):
    mask = torch.ones_like(batch_image_tensor)
    mask[:, :, top: bottom, left: right] = mask[:, :, top: bottom, left: right] - 1
    return mask * batch_image_tensor


def complement_tensor(batch_image_tensor, top, left, bottom, right):
    mask = torch.zeros_like(batch_image_tensor)
    mask[:, :, top: bottom, left: right] = mask[:, :, top: bottom, left: right] + 1
    return mask * batch_image_tensor


def crop_tensor_full(batch_image_tensor, top, left, bottom, right, value=0.):
    batch_image_tensor[:, :, top: bottom, left: right] = value
    return batch_image_tensor


def loss_decorator_crop(loss_function, top, left, bottom, right):
    def crop_loss(a, b):
        return loss_function(
            crop_tensor(a, top, left, bottom, right),
            crop_tensor(b, top, left, bottom, right)
        )
    return crop_loss


def loss_decorator_complement(loss_function, top, left, bottom, right):
    def crop_loss(a, b):
        return loss_function(
            complement_tensor(a, top, left, bottom, right),
            complement_tensor(b, top, left, bottom, right)
        )
    return crop_loss


def convert_array_to_images(np_array):
  """Converts numpy array to images with data type `uint8`.

  This function assumes the input numpy array is with range [-1, 1], as well as
  with shape [batch_size, channel, height, width]. Here, `channel = 3` for color
  image and `channel = 1` for gray image.

  The return images are with data type `uint8`, lying in range [0, 255]. In
  addition, the return images are with shape [batch_size, height, width,
  channel]. NOTE: the channel order will be the same as input.

  Inputs:
    np_array: The numpy array to convert.

  Returns:
    The converted images.

  Raises:
    ValueError: If this input is with wrong shape.
  """
  input_shape = np_array.shape
  if len(input_shape) != 4 or input_shape[1] not in [1, 3]:
    raise ValueError('Input `np_array` should be with shape [batch_size, '
                     'channel, height, width], where channel equals to 1 or 3. '
                     'But {} is received!'.format(input_shape))

  images = (np_array + 1.0) * 127.5
  images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
  images = images.transpose(0, 2, 3, 1)
  return images


def get_boundary(boundary_file_name):
    boundary = np.load(boundary_file_name, allow_pickle=True)
    a = boundary[()].reshape(1, 512)
    b = boundary[()]
    return a.astype(np.float32), b.astype(np.float32)


def get_interpolated_z(z, boundary_normal, max_step=3., num_frames=120):
    """
    :param z: (1, 512) shape
    :param boundary_normal: (1, 512) shape
    :return:
    """
    z_diff = boundary_normal * max_step
    l = np.linspace(0, 1, num_frames).reshape((num_frames, 1, 1))
    z_list = l * z_diff + z
    return z_list


def get_interpolated_wp(wp, boundary_normal, max_step=4., num_frames=120):
    """
    :param wp: (n, 512) shape
    :param boundary_normal: (1, 512) shape
    :return:
    """
    n = wp.shape[0]
    wp_diff = np.tile(boundary_normal, (n, 1)) * max_step
    l = np.linspace(0, 2, num_frames).reshape((num_frames, 1, 1))
    wp_list = l * wp_diff + wp - wp_diff
    return wp_list.astype(np.float32)


def interpolate_two_latent(latent_1, latent_2, step=10):
    assert len(latent_1) == len(latent_2), 'Check input latents'
    z_number = latent_1[0].shape[1]
    latent_list = []
    for i in range(len(latent_1)):
        dim_latent = len(latent_1[i].shape)
        latent_diff = latent_2[i][:, :z_number, :] - latent_1[i][:, :z_number, :]
        l = np.linspace(0, 1, step).reshape((step,) + (1,) * (dim_latent - 1))
        interpolated_list = l * latent_diff + latent_1[i][:, :z_number, :]
        latent_list.append(list(interpolated_list.astype(np.float32)))
    if len(latent_1) == 2:
        assert latent_list[0][0].shape[1] == latent_list[1][0].shape[1]
    return_list = list(zip(*latent_list))
    l2 = return_list[-1]
    return return_list


def plot_zs(input_list, pre_gan, post_gan):
    z_number = input_list[0].size()[1]
    images = []
    with torch.no_grad():
        for z_index in range(z_number):
            feature_before_alpha = pre_gan(input_list[0][:, z_index, :].view((-1, 512, 1, 1)))
            alphaed_feature = feature_before_alpha * input_list[1][:, z_index, :].view((-1, 512, 1, 1))
            image_tensor = post_gan(alphaed_feature)
            image_cv2 = convert_array_to_images(image_tensor.cpu().numpy())[0][:, :, ::-1]
            images.append(image_cv2)
    image_grid = get_grid(np.asarray(images), int(np.sqrt(z_number)), 256)
    return image_grid


def style_mixing(source_A, source_B, level):
    """
    :param source_A: of size [1, 18, 512]
    :param source_B: of size [1, 18, 512]
    :param level:
    :return:
    """
    assert level in LEVEL.keys(), 'Please Check Your Mixing Level.'
    start, end = LEVEL[level]
    new_latent = torch.zeros_like(source_A)
    new_latent[:, :, :] = source_A
    new_latent[:, start: end, :] = source_B[:, start: end, :]
    return new_latent


