# python 3.7
"""Contains the generator class of StyleGAN.

This class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import numpy as np

import torch

from . import model_settings
from .base_generator import BaseGenerator
from .stylegan_generator_network import StyleGANGeneratorNet

__all__ = ['StyleGANGenerator']


class StyleGANGenerator(BaseGenerator):
  """Defines the generator class of StyleGAN.

  Different from conventional GAN, StyleGAN introduces a disentangled latent
  space (i.e., W space) besides the normal latent space (i.e., Z space). Then,
  the disentangled latent code, w, is fed into each convolutional layer to
  modulate the `style` of the synthesis through AdaIN (Adaptive Instance
  Normalization) layer. Normally, the w's fed into all layers are the same. But,
  they can actually be different to make different layers get different styles.
  Accordingly, an extended space (i.e. W+ space) is used to gather all w's
  together. Taking the official StyleGAN model trained on FF-HQ dataset as an
  instance, there are
  (1) Z space, with dimension (512,)
  (2) W space, with dimension (512,)
  (3) W+ space, with dimension (18, 512)
  """

  def __init__(self, model_name, logger=None):
    super().__init__(model_name, logger)
    assert self.gan_type == 'stylegan'
    self.lod = self.net.synthesis.lod.to(self.cpu_device).tolist()
    self.logger.info(f'Current `lod` is {self.lod}.')

  def build(self):
    self.check_attr('w_space_dim')
    self.check_attr('fused_scale')
    self.truncation_psi = model_settings.STYLEGAN_TRUNCATION_PSI
    self.truncation_layers = model_settings.STYLEGAN_TRUNCATION_LAYERS
    self.randomize_noise = model_settings.STYLEGAN_RANDOMIZE_NOISE
    self.net = StyleGANGeneratorNet(
        resolution=self.resolution,
        z_space_dim=self.z_space_dim,
        w_space_dim=self.w_space_dim,
        image_channels=self.image_channels,
        fused_scale=self.fused_scale,
        truncation_psi=self.truncation_psi,
        truncation_layers=self.truncation_layers,
        randomize_noise=self.randomize_noise)
    self.num_layers = self.net.num_layers
    self.model_specific_vars = ['truncation.truncation']

  def convert_tf_weights(self, test_num=10):
    # pylint: disable=import-outside-toplevel
    import sys
    import pickle
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # pylint: enable=import-outside-toplevel

    sess = tf.compat.v1.InteractiveSession()

    self.logger.info(f'Loading tf weights from `{self.tf_weight_path}`.')
    self.check_attr('tf_code_path')
    sys.path.insert(0, self.tf_code_path)
    with open(self.tf_weight_path, 'rb') as f:
      _, _, tf_net = pickle.load(f)  # G, D, Gs
    sys.path.pop(0)
    self.logger.info(f'Successfully loaded!')

    self.logger.info(f'Converting tf weights to pytorch version.')
    tf_vars = dict(tf_net.__getstate__()['variables'])
    tf_vars.update(
        dict(tf_net.components.mapping.__getstate__()['variables']))
    tf_vars.update(
        dict(tf_net.components.synthesis.__getstate__()['variables']))
    state_dict = self.net.state_dict()
    for pth_var_name, tf_var_name in self.net.pth_to_tf_var_mapping.items():
      assert tf_var_name in tf_vars
      assert pth_var_name in state_dict
      self.logger.debug(f'  Converting `{tf_var_name}` to `{pth_var_name}`.')
      var = torch.from_numpy(np.array(tf_vars[tf_var_name]))
      if 'weight' in pth_var_name:
        if 'fc' in pth_var_name:
          var = var.permute(1, 0)
        elif 'conv' in pth_var_name:
          var = var.permute(3, 2, 0, 1)
      state_dict[pth_var_name] = var
    self.logger.info(f'Successfully converted!')

    self.logger.info(f'Saving pytorch weights to `{self.weight_path}`.')
    for var_name in self.model_specific_vars:
      del state_dict[var_name]
    torch.save(state_dict, self.weight_path)
    self.logger.info(f'Successfully saved!')

    self.load()

    # Start testing if needed.
    if test_num <= 0 or not tf.test.is_built_with_cuda():
      self.logger.warning(f'Skip testing the weights converted from tf model!')
      sess.close()
      return
    self.logger.info(f'Testing conversion results.')
    self.net.eval().to(self.run_device)
    total_distance = 0.0
    for i in range(test_num):
      latent_code = self.easy_sample(1)
      tf_output = tf_net.run(latent_code,  # latents_in
                             None,  # labels_in
                             truncation_psi=self.truncation_psi,
                             truncation_cutoff=self.truncation_layers,
                             randomize_noise=self.randomize_noise)
      pth_output = self.synthesize(latent_code)['image']
      distance = np.average(np.abs(tf_output - pth_output))
      self.logger.debug(f'  Test {i:03d}: distance {distance:.6e}.')
      total_distance += distance
    self.logger.info(f'Average distance is {total_distance / test_num:.6e}.')

    sess.close()

  def sample(self, num, latent_space_type='z', **kwargs):
    """Samples latent codes randomly.

    Args:
      num: Number of latent codes to sample. Should be positive.
      latent_space_type: Type of latent space from which to sample latent code.
        Only [`z`, `w`, `wp`] are supported. Case insensitive. (default: `z`)

    Returns:
      A `numpy.ndarray` as sampled latend codes.

    Raises:
      ValueError: If the given `latent_space_type` is not supported.
    """
    latent_space_type = latent_space_type.lower()
    if latent_space_type == 'z':
      latent_codes = np.random.randn(num, self.z_space_dim)
    elif latent_space_type in ['w', 'wp']:
      z = self.easy_sample(num, latent_space_type='z')
      latent_codes = []
      for inputs in self.get_batch_inputs(z, self.ram_size):
        outputs = self.easy_synthesize(latent_codes=inputs,
                                       latent_space_type='z',
                                       generate_style=False,
                                       generate_image=False)
        latent_codes.append(outputs[latent_space_type])
      latent_codes = np.concatenate(latent_codes, axis=0)
      if latent_space_type == 'w':
        assert latent_codes.shape == (num, self.w_space_dim)
      elif latent_space_type == 'wp':
        assert latent_codes.shape == (num, self.num_layers, self.w_space_dim)
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return latent_codes.astype(np.float32)

  def preprocess(self, latent_codes, latent_space_type='z', **kwargs):
    """Preprocesses the input latent code if needed.

    Args:
      latent_codes: The input latent codes for preprocessing.
      latent_space_type: Type of latent space to which the latent codes belong.
        Only [`z`, `w`, `wp`] are supported. Case insensitive. (default: `z`)

    Returns:
      The preprocessed latent codes which can be used as final input for the
        generator.

    Raises:
      ValueError: If the given `latent_space_type` is not supported.
    """
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    latent_space_type = latent_space_type.lower()
    if latent_space_type == 'z':
      latent_codes = latent_codes.reshape(-1, self.z_space_dim)
      norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
      latent_codes = latent_codes / norm * np.sqrt(self.z_space_dim)
    elif latent_space_type == 'w':
      latent_codes = latent_codes.reshape(-1, self.w_space_dim)
    elif latent_space_type == 'wp':
      latent_codes = latent_codes.reshape(-1, self.num_layers, self.w_space_dim)
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return latent_codes.astype(np.float32)

  def _synthesize(self,
                  latent_codes,
                  latent_space_type='z',
                  generate_style=False,
                  generate_image=True):
    """Synthesizes images with given latent codes.

    One can choose whether to generate the layer-wise style codes.

    Args:
      latent_codes: Input latent codes for image synthesis.
      latent_space_type: Type of latent space to which the latent codes belong.
        Only [`z`, `w`, `wp`] are supported. Case insensitive. (default: `z`)
      generate_style: Whether to generate the layer-wise style codes. (default:
        False)
      generate_image: Whether to generate the final image synthesis. (default:
        True)

    Returns:
      A dictionary whose values are raw outputs from the generator.
    """
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    results = {}

    latent_space_type = latent_space_type.lower()
    # Generate from Z space.
    if latent_space_type == 'z':
      if not (len(latent_codes.shape) == 2 and
              0 < latent_codes.shape[0] <= self.batch_size and
              latent_codes.shape[1] == self.z_space_dim):
        raise ValueError(f'Latent codes should be with shape [batch_size, '
                         f'latent_space_dim], where `batch_size` no larger '
                         f'than {self.batch_size}, and `latent_space_dim` '
                         f'equal to {self.z_space_dim}!\n'
                         f'But {latent_codes.shape} received!')
      zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      zs = zs.to(self.run_device)
      ws = self.net.mapping(zs)
      wps = self.net.truncation(ws)
      results['z'] = latent_codes
      results['w'] = self.get_value(ws)
      results['wp'] = self.get_value(wps)
    # Generate from W space.
    elif latent_space_type == 'w':
      if not (len(latent_codes.shape) == 2 and
              0 < latent_codes.shape[0] <= self.batch_size and
              latent_codes.shape[1] == self.w_space_dim):
        raise ValueError(f'Latent codes should be with shape [batch_size, '
                         f'w_space_dim], where `batch_size` no larger than '
                         f'{self.batch_size}, and `w_space_dim` equal to '
                         f'{self.w_space_dim}!\n'
                         f'But {latent_codes.shape} received!')
      ws = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      ws = ws.to(self.run_device)
      wps = self.net.truncation(ws)
      results['w'] = latent_codes
      results['wp'] = self.get_value(wps)
    # Generate from W+ space.
    elif latent_space_type == 'wp':
      if not (len(latent_codes.shape) == 3 and
              0 < latent_codes.shape[0] <= self.batch_size and
              latent_codes.shape[1] == self.num_layers and
              latent_codes.shape[2] == self.w_space_dim):
        raise ValueError(f'Latent codes should be with shape [batch_size, '
                         f'num_layers, w_space_dim], where `batch_size` no '
                         f'larger than {self.batch_size}, `num_layers` equal '
                         f'to {self.num_layers}, and `w_space_dim` equal to '
                         f'{self.w_space_dim}!\n'
                         f'But {latent_codes.shape} received!')
      wps = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      wps = wps.to(self.run_device)
      results['wp'] = latent_codes
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    if generate_style:
      for i in range(self.num_layers):
        style = self.net.synthesis.__getattr__(
            f'layer{i}').epilogue.style_mod.dense(wps[:, i, :])
        results[f'style{i:02d}'] = self.get_value(style)

    if generate_image:
      images = self.net.synthesis(wps)
      results['image'] = self.get_value(images)

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def synthesize(self,
                 latent_codes,
                 latent_space_type='z',
                 generate_style=False,
                 generate_image=True,
                 **kwargs):
    return self.batch_run(latent_codes,
                          lambda x: self._synthesize(
                              x,
                              latent_space_type=latent_space_type,
                              generate_style=generate_style,
                              generate_image=generate_image))
