#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Variational auto encoder for Stable Diffusion'''


# imports
import tensorflow as tf
#    script imports
# imports


# constants
DATA_FORMAT = 'channels_first'

# Used link to vizualize the conv blocks
# https://ezyang.github.io/convolution-visualizer/

# https://youtu.be/ZBKpAp_6TGI?si=stewpRrcYy-LsoVd&t=4550
# constants


# classes
class SiLU(tf.keras.layers.Layer):
  '''SiLU Block'''

  def __init__(self, beta=1.0):
    super().__init__()
    self.beta = beta

  def call(self, inputs):
    return inputs * tf.nn.sigmoid(self.beta * inputs)

# ------------------------------------------------------------------------------

class VAEResidualBlock(tf.keras.layers.Layer):
  '''Variational autoencoder - Residual block'''

  def __init__(self, out_channels:int=None):
    super().__init__()
    self.out_channels = out_channels

    self.group_norm_1 = tf.keras.layers.GroupNormalization(epsilon=1e-5)
    self.silu_01 = SiLU()
    self.conv_1 = tf.keras.layers.Conv2D(
      filters=out_channels,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding='same',
    )

    self.group_norm_2 = tf.keras.layers.GroupNormalization(epsilon=1e-5)
    self.silu_02 = SiLU()
    self.conv_2 = tf.keras.layers.Conv2D(
      filters=out_channels,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding='same',
    )

    self.conv_res_layer = tf.keras.layers.Conv2D(
      filters=out_channels,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='valid',
    )

    self.add = tf.keras.layers.Add()

    self.layer_list = [
      self.group_norm_1,
      self.silu_01,
      self.conv_1,
      self.group_norm_2,
      self.silu_02,
      self.conv_2
    ]

  def get_input_channels(self, inputs):
    return inputs.shape[-1]

  def call(self, inputs):
    x = inputs

    # holding onto residual values for residual layer
    residual = x

    # layer computations
    for module in self.layer_list:
      x = module(x)

    if self.get_input_channels(inputs) != self.out_channels:
      residual = self.conv_res_layer(residual)

    # adding residual value to the output

    x = self.add([x, residual])

    return x

# ------------------------------------------------------------------------------

class VAEAttentionBlock(tf.keras.layers.Layer):
  '''Variational autoencoder - Attention block'''

  def __init__(self, out_channels:int=None):
    super().__init__()
    # self.out_channels = out_channels
    self.group_norm_1 = tf.keras.layers.GroupNormalization(epsilon=1e-5)
    self.q = tf.keras.layers.Conv2D(
      filters=out_channels,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='valid',
    )

    self.k = tf.keras.layers.Conv2D(
      filters=out_channels,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='valid',
    )

    self.v = tf.keras.layers.Conv2D(
      filters=out_channels,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='valid',
    )
    self.proj_out = tf.keras.layers.Conv2D(
      filters=out_channels,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='valid',
    )

  def call(self, inputs):
    x = inputs

    h_ = self.group_norm_1(x)
    q, k, v = self.q(h_), self.k(h_), self.v(h_)

    # Compute attention
    b, h, w, c = q.shape
    q = tf.reshape(q, (-1, h * w, c))  # b,hw,c
    k = tf.keras.layers.Permute((3, 1, 2))(k)
    k = tf.reshape(k, (-1, c, h * w))  # b,c,hw
    w_ = q @ k
    w_ = w_ * (c ** (-0.5))
    w_ = tf.keras.activations.softmax(w_)

    # Attend to values
    v = tf.keras.layers.Permute((3, 1, 2))(v)
    v = tf.reshape(v, (-1, c, h * w))
    w_ = tf.keras.layers.Permute((2, 1))(w_)
    h_ = v @ w_
    h_ = tf.keras.layers.Permute((2, 1))(h_)
    h_ = tf.reshape(h_, (-1, h, w, c))
    return x + self.proj_out(h_)

# ------------------------------------------------------------------------------

class VAEEncoder(tf.keras.layers.Layer):
  '''Variational auto encoder - Encoder Block'''

  def __init__(self):
    super().__init__()
    self.conv_01 = tf.keras.layers.Conv2D(
      filters=128,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding='same',
    )
    self.res_block_01 = VAEResidualBlock(out_channels=128)
    self.res_block_02 = VAEResidualBlock(out_channels=128)

    self.conv_02 = tf.keras.layers.Conv2D(
      filters=128,
      kernel_size=(3, 3),
      strides=(2, 2),
      padding='valid',
    )

    self.res_block_03 = VAEResidualBlock(out_channels=256)
    self.res_block_04 = VAEResidualBlock(out_channels=256)

    self.conv_03 = tf.keras.layers.Conv2D(
      filters=256,
      kernel_size=(3, 3),
      strides=(2, 2),
      padding='valid',
    )

    self.res_block_05 = VAEResidualBlock(out_channels=512)
    self.res_block_06 = VAEResidualBlock(out_channels=512)

    self.conv_04 = tf.keras.layers.Conv2D(
      filters=512,
      kernel_size=(3, 3),
      strides=(2, 2),
      padding='valid',
    )

    self.res_block_07 = VAEResidualBlock(out_channels=512)
    self.res_block_08 = VAEResidualBlock(out_channels=512)
    self.res_block_09 = VAEResidualBlock(out_channels=512)

    self.attn_block_01 = VAEAttentionBlock(out_channels=512)

    self.res_block_10 = VAEResidualBlock(out_channels=512)

    self.group_norm_1 = tf.keras.layers.GroupNormalization(epsilon=1e-5)

    self.silu_01 = SiLU()

    self.conv_05 = tf.keras.layers.Conv2D(
      filters=8,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding='same',
    )

    self.conv_06 = tf.keras.layers.Conv2D(
      filters=8,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='valid',
    )

    self.layer_list = [
      self.conv_01,
      self.res_block_01,
      self.res_block_02,
      self.conv_02,
      self.res_block_03,
      self.res_block_04,
      self.conv_03,
      self.res_block_05,
      self.res_block_06,
      self.conv_04,
      self.res_block_07,
      self.res_block_08,
      self.res_block_09,
      self.attn_block_01,
      self.res_block_10,
      self.group_norm_1,
      self.silu_01,
      self.conv_05,
      self.conv_06,
    ]

  def call(self, inputs, noise):
    x = inputs

    for module in self.layer_list:
      if getattr(module, 'strides', None) == (2, 2):
        # Pad with zeros on the right and bottom.
        x = tf.pad(
          x,
          # (batch_size, height, width, channels)
          # adding pad to right and bottom of image
          paddings=tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]]),
          mode='CONSTANT',
          constant_values=0.0,
        )

      x = module(x)

    mean, log_variance = tf.split(x, 2, axis=3)
    stdev = tf.math.sqrt(
      tf.math.exp(
        tf.clip_by_value(log_variance, -30, 20)
      )
    )

    x = mean + stdev * noise

    # Scale by a constant
    # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
    # line 17: scale factor
    x *= 0.18215

    return x

# ------------------------------------------------------------------------------

class VAEDecoder(tf.keras.layers.Layer):
  '''Variational autoencoder - Decoder block'''

  def __init__(self):
    super().__init__()
    self.conv_01 = tf.keras.layers.Conv2D(
      filters=4,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='valid',
    )
    self.conv_02 = tf.keras.layers.Conv2D(
      filters=512,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding='same',
    )

    self.res_block_01 = VAEResidualBlock(out_channels=512)

    self.attn_block_01 = VAEAttentionBlock(out_channels=512)

    self.res_block_02 = VAEResidualBlock(out_channels=512)
    self.res_block_03 = VAEResidualBlock(out_channels=512)
    self.res_block_04 = VAEResidualBlock(out_channels=512)
    self.res_block_05 = VAEResidualBlock(out_channels=512)

    self.upsample_01 = tf.keras.layers.UpSampling2D(
      size=(2, 2),
    )

    self.conv_03 = tf.keras.layers.Conv2D(
      filters=512,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding='same',
    )

    self.res_block_06 = VAEResidualBlock(out_channels=512)
    self.res_block_07 = VAEResidualBlock(out_channels=512)
    self.res_block_08 = VAEResidualBlock(out_channels=512)

    self.upsample_02 = tf.keras.layers.UpSampling2D(
      size=(2, 2),
    )

    self.conv_04 = tf.keras.layers.Conv2D(
      filters=512,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding='same',
    )

    self.res_block_09 = VAEResidualBlock(out_channels=256)
    self.res_block_10 = VAEResidualBlock(out_channels=256)
    self.res_block_11 = VAEResidualBlock(out_channels=256)

    self.upsample_03 = tf.keras.layers.UpSampling2D(
      size=(2, 2),
    )

    self.conv_05 = tf.keras.layers.Conv2D(
      filters=256,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding='same',
    )

    self.res_block_12 = VAEResidualBlock(out_channels=128)
    self.res_block_13 = VAEResidualBlock(out_channels=128)
    self.res_block_14 = VAEResidualBlock(out_channels=128)

    self.group_norm_1 = tf.keras.layers.GroupNormalization(epsilon=1e-5)

    self.silu_01 = SiLU()

    self.conv_06 = tf.keras.layers.Conv2D(
      filters=3,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding='same',
    )

    self.layer_list = [
      self.conv_01,
      self.conv_02,
      self.res_block_01,
      self.attn_block_01,
      self.res_block_02,
      self.res_block_03,
      self.res_block_04,
      self.res_block_05,
      self.upsample_01,
      self.conv_03,
      self.res_block_06,
      self.res_block_07,
      self.res_block_08,
      self.upsample_02,
      self.conv_04,
      self.res_block_09,
      self.res_block_10,
      self.res_block_11,
      self.upsample_03,
      self.conv_05,
      self.res_block_12,
      self.res_block_13,
      self.res_block_14,
      self.group_norm_1,
      self.silu_01,
      self.conv_06
    ]

  def call(self, inputs):
    x = inputs

    x /= 0.18215

    for module in self.layer_list:
      x = module(x)

    return x
# classes


# functions
def test_vaeencoder():
  vaee = VAEEncoder()
  print(vaee(tf.random.uniform(shape=(32, 224, 224, 3))))
# functions


# main
def main():
  # test_vaeencoder()
  pass


# if main script
if __name__ == '__main__':
  main()
