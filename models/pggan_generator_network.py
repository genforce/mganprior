# python 3.7
"""Contains the implementation of generator described in PGGAN.

Different from the official tensorflow version in folder `pggan_tf_official`,
this is a simple pytorch version which only contains the generator part. This
class is specially used for inference.

For more details, please check the original paper:
https://arxiv.org/pdf/1710.10196.pdf
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PGGANGeneratorNet']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4


class PGGANGeneratorNet(nn.Module):
  """Defines the generator network in PGGAN.

  NOTE: The generated images are with `RGB` color channels and range [-1, 1].
  """

  def __init__(self,
               resolution=1024,
               z_space_dim=512,
               image_channels=3,
               fused_scale=False,
               fmaps_base=16 << 10,
               fmaps_max=512):
    """Initializes the generator with basic settings.

    Args:
      resolution: The resolution of the output image. (default: 1024)
      z_space_dim: The dimension of the initial latent space. (default: 512)
      image_channels: Number of channels of the output image. (default: 3)
      fused_scale: Whether to fused `upsample` and `conv2d` together, resulting
        in `conv2d_transpose`. (default: False)
      fmaps_base: Base factor to compute number of feature maps for each layer.
        (default: 16 << 10)
      fmaps_max: Maximum number of feature maps in each layer. (default: 512)

    Raises:
      ValueError: If the input `resolution` is not supported.
    """
    super().__init__()

    if resolution not in _RESOLUTIONS_ALLOWED:
      raise ValueError(f'Invalid resolution: {resolution}!\n'
                       f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')

    self.init_res = _INIT_RES
    self.init_res_log2 = int(np.log2(self.init_res))
    self.resolution = resolution
    self.final_res_log2 = int(np.log2(self.resolution))
    self.z_space_dim = z_space_dim
    self.image_channels = image_channels
    self.fused_scale = fused_scale
    self.fmaps_base = fmaps_base
    self.fmaps_max = fmaps_max

    self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

    self.lod = nn.Parameter(torch.zeros(()))
    self.pth_to_tf_var_mapping = {'lod': 'lod'}
    for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
      res = 2 ** res_log2
      block_idx = res_log2 - self.init_res_log2

      # First convolution layer for each resolution.
      if res == self.init_res:
        self.add_module(
            f'layer{2 * block_idx}',
            ConvBlock(in_channels=self.z_space_dim,
                      out_channels=self.get_nf(res),
                      kernel_size=self.init_res,
                      padding=3))
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.conv.weight'] = (
            f'{res}x{res}/Dense/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.wscale.bias'] = (
            f'{res}x{res}/Dense/bias')
      else:
        self.add_module(
            f'layer{2 * block_idx}',
            ConvBlock(in_channels=self.get_nf(res // 2),
                      out_channels=self.get_nf(res),
                      upsample=True,
                      fused_scale=self.fused_scale))
        if self.fused_scale:
          self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.weight'] = (
              f'{res}x{res}/Conv0_up/weight')
          self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.wscale.bias'] = (
              f'{res}x{res}/Conv0_up/bias')
        else:
          self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.conv.weight'] = (
              f'{res}x{res}/Conv0/weight')
          self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.wscale.bias'] = (
              f'{res}x{res}/Conv0/bias')

      # Second convolution layer for each resolution.
      self.add_module(
          f'layer{2 * block_idx + 1}',
          ConvBlock(in_channels=self.get_nf(res),
                    out_channels=self.get_nf(res)))
      if res == self.init_res:
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.conv.weight'] = (
            f'{res}x{res}/Conv/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.wscale.bias'] = (
            f'{res}x{res}/Conv/bias')
      else:
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.conv.weight'] = (
            f'{res}x{res}/Conv1/weight')
        self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.wscale.bias'] = (
            f'{res}x{res}/Conv1/bias')

      # Output convolution layer for each resolution.
      self.add_module(
          f'output{block_idx}',
          ConvBlock(in_channels=self.get_nf(res),
                    out_channels=self.image_channels,
                    kernel_size=1,
                    padding=0,
                    wscale_gain=1.0,
                    activation_type='linear'))
      self.pth_to_tf_var_mapping[f'output{block_idx}.conv.weight'] = (
          f'ToRGB_lod{self.final_res_log2 - res_log2}/weight')
      self.pth_to_tf_var_mapping[f'output{block_idx}.wscale.bias'] = (
          f'ToRGB_lod{self.final_res_log2 - res_log2}/bias')
    self.upsample = ResolutionScalingLayer()

  def get_nf(self, res):
    """Gets number of feature maps according to current resolution."""
    return min(self.fmaps_base // res, self.fmaps_max)

  def forward(self, z):
    if not (len(z.shape) == 2 and z.shape[1] == self.z_space_dim):
      raise ValueError(f'The input tensor should be with shape [batch_size, '
                       f'latent_space_dim], where `latent_space_dim` equals to '
                       f'{self.z_space_dim}!\n'
                       f'But {z.shape} received!')
    x = z.view(z.shape[0], self.z_space_dim, 1, 1)

    lod = self.lod.cpu().tolist()
    for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
      if res_log2 + lod <= self.final_res_log2:
        block_idx = res_log2 - self.init_res_log2
        x = self.__getattr__(f'layer{2 * block_idx}')(x)
        x = self.__getattr__(f'layer{2 * block_idx + 1}')(x)
        image = self.__getattr__(f'output{block_idx}')(x)
      else:
        image = self.upsample(image)
    return image


class PixelNormLayer(nn.Module):
  """Implements pixel-wise feature vector normalization layer."""

  def __init__(self, epsilon=1e-8):
    super().__init__()
    self.eps = epsilon

  def forward(self, x):
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


class ResolutionScalingLayer(nn.Module):
  """Implements the resolution scaling layer.

  Basically, this layer can be used to upsample feature maps from spatial domain
  with nearest neighbor interpolation.
  """

  def __init__(self, scale_factor=2):
    super().__init__()
    self.scale_factor = scale_factor

  def forward(self, x):
    return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class WScaleLayer(nn.Module):
  """Implements the layer to scale weight variable and add bias.

  NOTE: The weight variable is trained in `nn.Conv2d` layer, and only scaled
  with a constant number, which is not trainable in this layer. However, the
  bias variable is trainable in this layer.
  """

  def __init__(self, in_channels, out_channels, kernel_size, gain=np.sqrt(2.0)):
    super().__init__()
    fan_in = in_channels * kernel_size * kernel_size
    self.scale = gain / np.sqrt(fan_in)
    self.bias = nn.Parameter(torch.zeros(out_channels))

  def forward(self, x):
    return x * self.scale + self.bias.view(1, -1, 1, 1)


class ConvBlock(nn.Module):
  """Implements the convolutional block.

  Basically, this block executes pixel-wise normalization layer, upsampling
  layer (if needed), convolutional layer, weight-scale layer, and activation
  layer in sequence.
  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               dilation=1,
               add_bias=False,
               upsample=False,
               fused_scale=False,
               wscale_gain=np.sqrt(2.0),
               activation_type='lrelu'):
    """Initializes the class with block settings.

    Args:
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels of the output tensor.
      kernel_size: Size of the convolutional kernels.
      stride: Stride parameter for convolution operation.
      padding: Padding parameter for convolution operation.
      dilation: Dilation rate for convolution operation.
      add_bias: Whether to add bias onto the convolutional result.
      upsample: Whether to upsample the input tensor before convolution.
      fused_scale: Whether to fused `upsample` and `conv2d` together, resulting
        in `conv2d_transpose`.
      wscale_gain: The gain factor for `wscale` layer.
      activation_type: Type of activation function. Support `linear`, `lrelu`
        and `tanh`.

    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
    super().__init__()

    self.pixel_norm = PixelNormLayer()

    if upsample and not fused_scale:
      self.upsample = ResolutionScalingLayer()
    else:
      self.upsample = nn.Identity()

    if upsample and fused_scale:
      self.use_conv2d_transpose = True
      self.weight = nn.Parameter(
          torch.randn(kernel_size, kernel_size, in_channels, out_channels))
      fan_in = in_channels * kernel_size * kernel_size
      self.scale = wscale_gain / np.sqrt(fan_in)
    else:
      self.use_conv2d_transpose = False
      self.conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=1,
                            bias=add_bias)

    self.wscale = WScaleLayer(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              gain=wscale_gain)

    if activation_type == 'linear':
      self.activate = nn.Identity()
    elif activation_type == 'lrelu':
      self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif activation_type == 'tanh':
      self.activate = nn.Hardtanh()
    else:
      raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

  def forward(self, x):
    x = self.pixel_norm(x)
    x = self.upsample(x)
    if self.use_conv2d_transpose:
      kernel = self.weight * self.scale
      kernel = F.pad(kernel, (0, 0, 0, 0, 1, 1, 1, 1), 'constant', 0.0)
      kernel = (kernel[1:, 1:] + kernel[:-1, 1:] +
                kernel[1:, :-1] + kernel[:-1, :-1])
      kernel = kernel.permute(2, 3, 0, 1)
      x = F.conv_transpose2d(x, kernel, stride=2, padding=1)
      x = x / self.scale
    else:
      x = self.conv(x)
    x = self.wscale(x)
    x = self.activate(x)
    return x
