# python 3.7
"""Contains the base class for generator in a GAN model."""

import os.path
import sys
import logging
import numpy as np

import torch

from . import model_settings

__all__ = ['BaseGenerator']


def get_temp_logger(logger_name='logger'):
  """Gets a temporary logger.

  This logger will print all levels of messages onto the screen.

  Args:
    logger_name: Name of the logger.

  Returns:
    A `logging.Logger`.

  Raises:
    ValueError: If the input `logger_name` is empty.
  """
  if not logger_name:
    raise ValueError(f'Input `logger_name` should not be empty!')

  logger = logging.getLogger(logger_name)
  if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

  return logger


class BaseGenerator(object):
  """Base class for generator used in GAN variants."""

  def __init__(self, model_name, logger=None):
    """Initializes with specific settings.

    The GAN model should be first registered in `model_settings.py` with proper
    settings. Among them, some attributes are necessary, including:

    (1) gan_type: Type of the GAN model.
    (2) z_space_dim: Dimension of the latent space.
    (3) resolution: Resolution of the synthesis.
    (4) min_val: Minimum value of the raw synthesis. (default -1.0)
    (5) max_val: Maximum value of the raw synthesis. (default 1.0)
    (6) image_channels: Number of channels of the synthesis. (default: 3)
    (7) channel_order: Channel order of the raw synthesis. (default: `RGB`)

    Args:
      model_name: Name with which the GAN model is registered.
      logger: Logger for recording log messages. If set as `None`, a default
        logger, which prints messages from all levels onto the screen, will be
        created. (default: None)

    Raises:
      AttributeError: If some necessary attributes are missing.
    """
    self.model_name = model_name
    self.logger = logger or get_temp_logger(model_name)

    # Parse settings.
    for key, val in model_settings.MODEL_POOL[model_name].items():
      setattr(self, key, val)
    self.use_cuda = model_settings.USE_CUDA and torch.cuda.is_available()
    self.batch_size = model_settings.MAX_IMAGES_ON_DEVICE
    self.ram_size = model_settings.MAX_IMAGES_ON_RAM
    self.net = None
    self.run_device = 'cuda' if self.use_cuda else 'cpu'
    self.cpu_device = 'cpu'

    # Check necessary settings.
    self.weight_path = getattr(self, 'weight_path', '')
    self.tf_weight_path = getattr(self, 'tf_weight_path', '')
    self.check_attr('gan_type')
    self.check_attr('z_space_dim')
    self.check_attr('resolution')
    self.min_val = getattr(self, 'min_val', -1.0)
    self.max_val = getattr(self, 'max_val', 1.0)
    self.image_channels = getattr(self, 'image_channels', 3)
    assert self.image_channels in [1, 3]
    self.channel_order = getattr(self, 'channel_order', 'RGB').upper()
    assert self.channel_order in ['RGB', 'BGR']

    # Build graph and load pre-trained weights.
    self.logger.info(f'Build generator for model `{self.model_name}`.')
    self.model_specific_vars = []
    self.build()
    if os.path.isfile(self.weight_path):
      self.load()
    elif os.path.isfile(self.tf_weight_path):
      self.convert_tf_weights()
    else:
      self.logger.warning(f'No pre-trained weights will be loaded!')

    # Change to inference mode and GPU mode if needed.
    assert self.net
    self.net.eval().to(self.run_device)

  def check_attr(self, attr_name):
    """Checks the existence of a particular attribute.

    Args:
      attr_name: Name of the attribute to check.

    Raises:
      AttributeError: If the target attribute is missing.
    """
    if not hasattr(self, attr_name):
      raise AttributeError(f'Field `{attr_name}` is missing for '
                           f'generator in model `{self.model_name}`!')

  def build(self):
    """Builds the graph."""
    raise NotImplementedError(f'Should be implemented in derived class!')

  def load(self):
    """Loads pre-trained weights."""
    self.logger.info(f'Loading pytorch weights from `{self.weight_path}`.')
    state_dict = torch.load(self.weight_path)
    for var_name in self.model_specific_vars:
      state_dict[var_name] = self.net.state_dict()[var_name]
    self.net.load_state_dict(state_dict)
    self.logger.info(f'Successfully loaded!')

  def convert_tf_weights(self, test_num=10):
    """Converts weights from tensorflow version.

    Args:
      test_num: Number of samples used for testing whether the conversion is
        done correctly. `0` disables the test. (default: 10)
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def get_value(self, tensor):
    """Gets value of a `torch.Tensor`.

    Args:
      tensor: The input tensor to get value from.

    Returns:
      A `numpy.ndarray`.

    Raises:
      ValueError: If the tensor is with neither `torch.Tensor` type or
        `numpy.ndarray` type.
    """
    dtype = type(tensor)
    if isinstance(tensor, np.ndarray):
      return tensor
    if isinstance(tensor, torch.Tensor):
      return tensor.to(self.cpu_device).detach().numpy()
    raise ValueError(f'Unsupported input type `{dtype}`!')

  def get_batch_inputs(self, inputs, batch_size=None):
    """Gets inputs within mini-batch.

    This function yields at most `self.batch_size` inputs at a time.

    Args:
      inputs: Input data to form mini-batch.
      batch_size: Batch size. If not specified, `self.batch_size` will be used.
        (default: None)
    """
    total_num = inputs.shape[0]
    batch_size = batch_size or self.batch_size
    for i in range(0, total_num, batch_size):
      yield inputs[i:i + batch_size]

  def batch_run(self, inputs, run_fn):
    """Runs model with mini-batch.

    This function splits the inputs into mini-batches, run the model with each
    mini-batch, and then concatenate the outputs from all mini-batches together.

    NOTE: The output of `run_fn` can only be `numpy.ndarray` or a dictionary
    whose values are all `numpy.ndarray`.

    Args:
      inputs: The input samples to run with.
      run_fn: A callable function.

    Returns:
      Same type as the output of `run_fn`.

    Raises:
      ValueError: If the output type of `run_fn` is not supported.
    """
    if inputs.shape[0] > self.ram_size:
      self.logger.warning(f'Number of inputs on RAM is larger than '
                          f'{self.ram_size}. Please use '
                          f'`self.get_batch_inputs()` to split the inputs! '
                          f'Otherwise, it may encounter OOM problem!')

    results = {}
    temp_key = '__temp_key__'
    for batch_inputs in self.get_batch_inputs(inputs):
      batch_outputs = run_fn(batch_inputs)
      if isinstance(batch_outputs, dict):
        for key, val in batch_outputs.items():
          if not isinstance(val, np.ndarray):
            raise ValueError(f'Each item of the model output should be with '
                             f'type `numpy.ndarray`, but type `{type(val)}` is '
                             f'received for key `{key}`!')
          if key not in results:
            results[key] = [val]
          else:
            results[key].append(val)
      elif isinstance(batch_outputs, np.ndarray):
        if temp_key not in results:
          results[temp_key] = [batch_outputs]
        else:
          results[temp_key].append(batch_outputs)
      else:
        raise ValueError(f'The model output can only be with type '
                         f'`numpy.ndarray`, or a dictionary of '
                         f'`numpy.ndarray`, but type `{type(batch_outputs)}` '
                         f'is received!')

    for key, val in results.items():
      results[key] = np.concatenate(val, axis=0)
    return results if temp_key not in results else results[temp_key]

  def sample(self, num, **kwargs):
    """Samples latent codes randomly.

    Args:
      num: Number of latent codes to sample. Should be positive.

    Returns:
      A `numpy.ndarray` as sampled latend codes.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def preprocess(self, latent_codes, **kwargs):
    """Preprocesses the input latent codes if needed.

    Args:
      latent_codes: The input latent codes for preprocessing.

    Returns:
      The preprocessed latent codes which can be used as final inputs to the
        generator.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def easy_sample(self, num, **kwargs):
    """Wraps functions `sample()` and `preprocess()` together."""
    return self.preprocess(self.sample(num, **kwargs), **kwargs)

  def synthesize(self, latent_codes, **kwargs):
    """Synthesizes images with given latent codes.

    NOTE: The latent codes are assumed to have already been preprocessed.

    Args:
      latent_codes: Input latent codes for image synthesis.

    Returns:
      A dictionary whose values are raw outputs from the generator. Keys of the
        dictionary usually include `z` and `image`.
    """
    raise NotImplementedError(f'Should be implemented in derived class!')

  def postprocess(self, images):
    """Postprocesses the output images if needed.

    This function assumes the input numpy array is with shape [batch_size,
    channel, height, width]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. The returned images are with shape
    [batch_size, height, width, channel].

    NOTE: The channel order of output images will always be `RGB`.

    Args:
      images: The raw outputs from the generator.

    Returns:
      The postprocessed images with dtype `numpy.uint8` and range [0, 255].

    Raises:
      ValueError: If the input `images` are not with type `numpy.ndarray` or not
        with shape [batch_size, channel, height, width].
    """
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')

    if len(images.shape) != 4 or images.shape[1] not in [1, 3]:
      raise ValueError(f'Input should be with shape [batch_size, channel, '
                       f'height, width], where channel equals to 1 or 3!\n'
                       f'But {images.shape} is received!')
    assert images.shape[1] == self.image_channels
    images = (images - self.min_val) * 255 / (self.max_val - self.min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    if self.image_channels == 3 and self.channel_order == 'BGR':
      images = images[:, :, :, ::-1]

    return images

  def easy_synthesize(self, latent_codes, **kwargs):
    """Wraps functions `synthesize()` and `postprocess()` together."""
    outputs = self.synthesize(latent_codes, **kwargs)
    if 'image' in outputs:
      outputs['image'] = self.postprocess(outputs['image'])
    return outputs
