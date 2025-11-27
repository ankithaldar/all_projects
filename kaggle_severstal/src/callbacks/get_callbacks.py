#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports

from dataclasses import dataclass

from helpers.module_importer import instantiate_class

# imports


# constants
# constants


# classes
@dataclass
class Callback:
  class_path: str
  init_args: dict


def get_callbacks():
  return [
    Callback(
      class_path='pytorch_lightning.callbacks.EarlyStopping',
      init_args={'monitor': 'val_F1Score', 'patience': 5, 'mode': 'min', 'min_delta': 0.0, 'check_finite': True, 'check_on_train_epoch_end': False}
    ),
    Callback(
      class_path='pytorch_lightning.callbacks.LearningRateMonitor',
      init_args={'logging_interval': 'step', 'log_momentum': True}
    ),
    Callback(
      class_path='pytorch_lightning.callbacks.ModelCheckpoint',
      init_args={'dirpath': './checkpoints', 'filename': "kaggle_severstal_efficientnet_b1_01_{epoch:02d}-{step}-{val_F1Score:.3f}", 'monitor': 'val_F1Score', 'mode': 'min', 'save_top_k': 3, 'save_last': True, 'save_weights_only': False, 'auto_insert_metric_name': True, 'every_n_epochs': None, 'train_time_interval': None}
    ),
    Callback(
      class_path='callbacks.discord_callback.DiscordCallback',
      init_args={'experiment_name': 'kaggle_severstal_efficientnet_b1_01', 'log_every_n_steps': 200}
    )
  ]

class Callbacks:
  '''Callbacks'''

  def __init__(self):
    self.callback_list = get_callbacks()

  def __call__(self):
    callbacks = []
    if isinstance(self.callback_list, list) and len(self.callback_list) > 0:
      callbacks = [
        instantiate_class(callback.class_path, callback.init_args)
        for callback in self.callback_list
      ]

    return callbacks

# classes
