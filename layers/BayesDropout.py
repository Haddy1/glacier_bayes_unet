#import keras.backend as K
from tensorflow.python.keras import backend as K
from keras.layers import Layer, Dropout
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn


class BayesDropout(Dropout):

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)

  def call(self, inputs, training=None):

    return nn.dropout(
          inputs,
          noise_shape=self._get_noise_shape(inputs),
          seed=self.seed,
          rate=self.rate)



