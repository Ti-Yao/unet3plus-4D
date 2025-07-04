# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Layer utilities."""

import tensorflow as tf
from array_ops import resize_with_crop_or_pad

from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv1D, Conv3D, MaxPooling1D, MaxPooling3D, UpSampling1D, UpSampling3D, TimeDistributed, Lambda)
import tensorflow as tf



def Transpose(perm):
    return Lambda(lambda x: tf.transpose(x, perm))


def Multi_TimeDistributed(layer, iter):
    for _ in range(iter):
        layer = TimeDistributed(layer)
    return layer



class Conv4D(layers.Layer):
    def __init__(self, 
                 filters,
                 kernel_size,
                 padding="same",
                 kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
                 bias_initializer=tf.keras.initializers.Zeros(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activation=None,
                 **kwargs):  # Accept other keyword args like 'name'
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activation = activation

        self.conv3d_1 = layers.TimeDistributed(
            layers.Conv3D(filters, 
                          kernel_size, 
                          padding=padding,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          activation=activation
                          )
        )
        
        self.conv1d_1 = Multi_TimeDistributed(
            layers.Conv1D(filters, 
                          kernel_size[0], 
                          padding=padding,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          activation=activation
                          ),
            iter=3
        )

    def call(self, inputs):
        x = self.conv3d_1(inputs)
        # Use tf.transpose instead of Transpose layer
        x = tf.transpose(x, perm=(0, 2, 3, 4, 1, 5))
        x = self.conv1d_1(x)
        x = tf.transpose(x, perm=(0, 4, 1, 2, 3, 5))
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "activation": self.activation
        })
        return config




class MaxPool4D(layers.Layer):
    def __init__(self, 
                 pool_size,
                 **kwargs):  # Accept other keyword args like 'name'
                 
        super().__init__()
        self.pool_size = pool_size
        self.pool3d = TimeDistributed(MaxPooling3D(pool_size = pool_size[1:]))
        self.time_pool = (pool_size[0],)

        self.pool1d = Multi_TimeDistributed(MaxPooling1D(pool_size = self.time_pool), iter=3)

    def call(self, inputs):
        p = self.pool3d(inputs)
        if self.time_pool[0] > 1:
          p = Transpose((0, 2, 3, 4, 1, 5))(p)
          p = self.pool1d(p)
          p = Transpose((0, 4, 1, 2, 3, 5))(p)
        return p


    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size
        })
        return config
    
class UpSampling4D(layers.Layer):
    def __init__(self, 
                 size,
                 **kwargs):  # Accept other keyword args like 'name'
                 
        super().__init__()
        self.size = size

        self.upsample3d = TimeDistributed(UpSampling3D(size = size[1:]))
        self.time_size = size[0]
        self.upsample1d = Multi_TimeDistributed(UpSampling1D(size = self.time_size), iter=3)

    def call(self, inputs):
        x = self.upsample3d(inputs)
        if self.time_size > 1:
          x = Transpose((0, 2, 3, 4, 1, 5))(x)
          x = self.upsample1d(x)
          x = Transpose((0, 4, 1, 2, 3, 5))(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "size": self.size
        })
        return config



def get_nd_layer(name, rank):
  """Get an N-D layer object.

  Args:
    name: A `str`. The name of the requested layer.
    rank: An `int`. The rank of the requested layer.

  Returns:
    A `tf.keras.layers.Layer` object.

  Raises:
    ValueError: If the requested layer is unknown to TFMRI.
  """
  try:
    return _ND_LAYERS[(name, rank)]
  except KeyError as err:
    raise ValueError(
        f"Could not find a layer with name '{name}' and rank {rank}.") from err


_ND_LAYERS = {
    ('AveragePooling', 1): tf.keras.layers.AveragePooling1D,
    ('AveragePooling', 2): tf.keras.layers.AveragePooling2D,
    ('AveragePooling', 3): tf.keras.layers.AveragePooling3D,
    ('Conv', 1): tf.keras.layers.Conv1D,
    ('Conv', 2): tf.keras.layers.Conv2D,
    ('Conv', 3): tf.keras.layers.Conv3D,
    ('Conv', 4): Conv4D,
    ('ConvLSTM', 1): tf.keras.layers.ConvLSTM1D,
    ('ConvLSTM', 2): tf.keras.layers.ConvLSTM2D,
    ('ConvLSTM', 3): tf.keras.layers.ConvLSTM3D,
    ('ConvTranspose', 1): tf.keras.layers.Conv1DTranspose,
    ('ConvTranspose', 2): tf.keras.layers.Conv2DTranspose,
    ('ConvTranspose', 3): tf.keras.layers.Conv3DTranspose,
    ('Cropping', 1): tf.keras.layers.Cropping1D,
    ('Cropping', 2): tf.keras.layers.Cropping2D,
    ('Cropping', 3): tf.keras.layers.Cropping3D,
    ('DepthwiseConv', 1): tf.keras.layers.DepthwiseConv1D,
    ('DepthwiseConv', 2): tf.keras.layers.DepthwiseConv2D,
    ('GlobalAveragePooling', 1): tf.keras.layers.GlobalAveragePooling1D,
    ('GlobalAveragePooling', 2): tf.keras.layers.GlobalAveragePooling2D,
    ('GlobalAveragePooling', 3): tf.keras.layers.GlobalAveragePooling3D,
    ('GlobalMaxPool', 1): tf.keras.layers.GlobalMaxPool1D,
    ('GlobalMaxPool', 2): tf.keras.layers.GlobalMaxPool2D,
    ('GlobalMaxPool', 3): tf.keras.layers.GlobalMaxPool3D,
    ('MaxPool', 1): tf.keras.layers.MaxPool1D,
    ('MaxPool', 2): tf.keras.layers.MaxPool2D,
    ('MaxPool', 3): tf.keras.layers.MaxPool3D,
    ('MaxPool', 4): MaxPool4D,
    
    ('SeparableConv', 1): tf.keras.layers.SeparableConv1D,
    ('SeparableConv', 2): tf.keras.layers.SeparableConv2D,
    ('SpatialDropout', 1): tf.keras.layers.SpatialDropout1D,
    ('SpatialDropout', 2): tf.keras.layers.SpatialDropout2D,
    ('SpatialDropout', 3): tf.keras.layers.SpatialDropout3D,
    ('UpSampling', 1): tf.keras.layers.UpSampling1D,
    ('UpSampling', 2): tf.keras.layers.UpSampling2D,
    ('UpSampling', 3): tf.keras.layers.UpSampling3D,
    ('UpSampling', 4): UpSampling4D,
    ('ZeroPadding', 1): tf.keras.layers.ZeroPadding1D,
    ('ZeroPadding', 2): tf.keras.layers.ZeroPadding2D,
    ('ZeroPadding', 3): tf.keras.layers.ZeroPadding3D
}


class ResizeAndConcatenate(tf.keras.layers.Layer):
  """Resizes and concatenates a list of inputs.

  Similar to `tf.keras.layers.Concatenate`, but if the inputs have different
  shapes, they are resized to match the shape of the first input.

  Args:
    axis: Axis along which to concatenate.
  """
  def __init__(self, axis=-1, **kwargs):
    super().__init__(**kwargs)
    self.axis = axis

  def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
        })
        return config

  def call(self, inputs):  
    if not isinstance(inputs, (list, tuple)):
      raise ValueError(
          f"Layer {self.__class__.__name__} expects a list of inputs. "
          f"Received: {inputs}")

    rank = inputs[0].shape.rank
    if rank is None:
      raise ValueError(
          f"Layer {self.__class__.__name__} expects inputs with known rank. "
          f"Received: {inputs}")
    if self.axis >= rank or self.axis < -rank:
      raise ValueError(
          f"Layer {self.__class__.__name__} expects `axis` to be in the range "
          f"[-{rank}, {rank}) for an input of rank {rank}. "
          f"Received: {self.axis}")
    # Canonical axis (always positive).
    axis = self.axis % rank

    # Resize inputs.
    shape = tf.tensor_scatter_nd_update(tf.shape(inputs[0]), [[axis]], [-1])
    resized = [resize_with_crop_or_pad(tensor, shape)
               for tensor in inputs[1:]]

    # Set the static shape for each resized tensor.
    for i, tensor in enumerate(resized):
      static_shape = inputs[0].shape.as_list()
      static_shape[axis] = inputs[i + 1].shape.as_list()[axis]
      static_shape = tf.TensorShape(static_shape)
      resized[i] = tf.ensure_shape(tensor, static_shape)
    return tf.concat(inputs[:1] + resized, axis=self.axis)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
