#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Alpha Zero from Scratch - model'''


# imports
import tensorflow as tf
#    script imports
# imports


# constants
# constants


# classes
class ResNetStartBlock(tf.keras.layers.Layer):
  '''Resnet model start layer'''

  def __init__(self, num_hidden):
    super().__init__()
    self.conv_block = tf.keras.layers.Conv2D(
      filters=num_hidden,
      kernel_size=(3, 3),
      padding='same'
    )

    self.batch_norm = tf.keras.layers.BatchNormalization()

    self.relu = tf.keras.layers.ReLU()

    # create a list of all layers in order for to run through for loop
    self.layer_list = [self.conv_block, self.batch_norm, self.relu]

  def call(self, inputs):
    x = inputs

    for layer in self.layer_list:
      x = layer(x)

    return x



class ResNetResBlock(tf.keras.layers.Layer):
  '''Resnet model Residual Blocks'''

  def __init__(self, num_hidden):
    super().__init__()
    self.conv_block_01 = tf.keras.layers.Conv2D(
      filters=num_hidden,
      kernel_size=(3, 3),
      padding='same'
    )

    self.batch_norm_01 = tf.keras.layers.BatchNormalization()

    self.relu_01 = tf.keras.layers.ReLU()

    self.conv_block_02 = tf.keras.layers.Conv2D(
      filters=num_hidden,
      kernel_size=(3, 3),
      padding='same'
    )

    self.batch_norm_02 = tf.keras.layers.BatchNormalization()

    # to use relu after residual concat
    self.relu_02 = tf.keras.layers.ReLU()

    # create a list of all layers in order for to run through for loop
    self.layer_list = [self.conv_block_01, self.batch_norm_01, self.relu_01, self.conv_block_02, self.batch_norm_02]

  def call(self, inputs):
    x = inputs
    residual = x

    for layer in self.layer_list:
      x = layer(x)

    x += residual

    x = self.relu_02(x)

    return x



class ResNetPolicyHead(tf.keras.layers.Layer):
  '''Resnet model Policy Head'''

  def __init__(self, game):
    super().__init__()
    self.conv_block = tf.keras.layers.Conv2D(
      filters=32,
      kernel_size=(3, 3),
      padding='same'
    )

    self.batch_norm = tf.keras.layers.BatchNormalization()

    self.relu = tf.keras.layers.ReLU()

    self.flatten = tf.keras.layers.Flatten()

    self.dense = tf.keras.layers.Dense(
      units=game.action_size,
      activation=None
    )

    # create a list of all layers in order for to run through for loop
    self.layer_list = [self.conv_block, self.batch_norm, self.relu, self.flatten, self.dense]

  def call(self, inputs):
    x = inputs

    for layer in self.layer_list:
      x = layer(x)

    return x



class ResNetValueHead(tf.keras.layers.Layer):
  '''Resnet model Value Head'''

  def __init__(self):
    super().__init__()
    self.conv_block = tf.keras.layers.Conv2D(
      filters=3,
      kernel_size=(3, 3),
      padding='same'
    )

    self.batch_norm = tf.keras.layers.BatchNormalization()

    self.relu = tf.keras.layers.ReLU()

    self.flatten = tf.keras.layers.Flatten()

    self.dense = tf.keras.layers.Dense(
      units=1
    )

    # create a list of all layers in order for to run through for loop
    self.layer_list = [self.conv_block, self.batch_norm, self.relu, self.flatten, self.dense]

  def call(self, inputs):
    x = inputs

    for layer in self.layer_list:
      x = layer(x)

    x = tf.keras.activations.tanh(x)

    return x



class ResNet(tf.keras.Model):
  '''ResNet model'''

  def __init__(self, game, num_res_blocks, num_hidden):
    super().__init__()
    self.start_block = ResNetStartBlock(num_hidden=num_hidden)

    self.backbone = [ResNetResBlock(num_hidden) for _ in range(num_res_blocks)]

    self.policy_head = ResNetPolicyHead(game)

    self.value_head = ResNetValueHead()


    # create a list of all layers in order for to run through for loop
    self.layer_list = [self.start_block, *self.backbone]

  def call(self, inputs):
    x = inputs

    for layer in self.layer_list:
      x = layer(x)

    # returning in order of policy and value
    return self.policy_head(x), self.value_head(x)

# classes


# functions
def function_name():
  pass
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
