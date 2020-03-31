# python 3.7
"""Contains the generator class of PGGAN.

This class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import numpy as np

import torch

from .base_generator import BaseGenerator
from .pggan_generator_network import PGGANGeneratorNet

__all__ = ['PGGANGenerator']


class PGGANGenerator(BaseGenerator):
  """Defines the generator class of PGGAN."""

  def __init__(self, model_name, logger=None):
    super().__init__(model_name, logger)
    assert self.gan_type == 'pggan'
    self.lod = self.net.lod.to(self.cpu_device).tolist()
    self.logger.info(f'Current `lod` is {self.lod}.')

  def build(self):
    self.check_attr('fused_scale')
    self.net = PGGANGeneratorNet(resolution=self.resolution,
                                 z_space_dim=self.z_space_dim,
                                 image_channels=self.image_channels,
                                 fused_scale=self.fused_scale)
    self.num_layers = self.net.num_layers

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
    state_dict = self.net.state_dict()
    for pth_var_name, tf_var_name in self.net.pth_to_tf_var_mapping.items():
      assert tf_var_name in tf_vars
      assert pth_var_name in state_dict
      self.logger.debug(f'  Converting `{tf_var_name}` to `{pth_var_name}`.')
      var = torch.from_numpy(np.array(tf_vars[tf_var_name]))
      if 'weight' in pth_var_name:
        if 'layer0.conv' in pth_var_name:
          var = var.view(var.shape[0], -1, self.net.init_res, self.net.init_res)
          var = var.permute(1, 0, 2, 3).flip(2, 3)
        elif 'conv' in pth_var_name:
          var = var.permute(3, 2, 0, 1)
        elif 'conv' not in pth_var_name:
          var = var.permute(0, 1, 3, 2)
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
    label_dim = tf_net.input_shapes[1][1]
    tf_fake_label = np.zeros((1, label_dim), np.float32)
    total_distance = 0.0
    for i in range(test_num):
      latent_code = self.easy_sample(1)
      tf_output = tf_net.run(latent_code, tf_fake_label)
      pth_output = self.synthesize(latent_code)['image']
      distance = np.average(np.abs(tf_output - pth_output))
      self.logger.debug(f'  Test {i:03d}: distance {distance:.6e}.')
      total_distance += distance
    self.logger.info(f'Average distance is {total_distance / test_num:.6e}.')

    sess.close()

  def sample(self, num, **kwargs):
    assert num > 0
    return np.random.randn(num, self.z_space_dim).astype(np.float32)

  def preprocess(self, latent_codes, **kwargs):
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    latent_codes = latent_codes.reshape(-1, self.z_space_dim)
    norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
    latent_codes = latent_codes / norm * np.sqrt(self.z_space_dim)
    return latent_codes.astype(np.float32)

  def _synthesize(self, latent_codes):
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')
    if not (len(latent_codes.shape) == 2 and
            0 < latent_codes.shape[0] <= self.batch_size and
            latent_codes.shape[1] == self.z_space_dim):
      raise ValueError(f'Latent codes should be with shape [batch_size, '
                       f'latent_space_dim], where `batch_size` no larger than '
                       f'{self.batch_size}, and `latent_space_dim` equals to '
                       f'{self.z_space_dim}!\n'
                       f'But {latent_codes.shape} received!')

    zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
    zs = zs.to(self.run_device)
    images = self.net(zs)
    results = {
        'z': latent_codes,
        'image': self.get_value(images),
    }

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def synthesize(self, latent_codes, **kwargs):
    return self.batch_run(latent_codes, self._synthesize)
