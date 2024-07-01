#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Base Engine for model training'''


# imports
import tensorflow as tf
#    script imports
# imports


# constants
# constants


# classes
class BaseEngine:
  '''Base Engine for model training'''

  def __init__(self, hparams):
    self.hparams = hparams

    self._init_score_function()
    self._init_augmentation()

    self._init_train_datalader()
    self._init_valid_dataloader()
    self._init_test_dataloader()

    self._init_model()
    self._init_loss_function()
    self._init_metrics()

    self.setup()


  def _init_score_function(self):
    # look into this from kaggle-birdsong-recognition repo
    pass


  def _init_augmentation(self):
    self.tfms = None


  def _init_train_dataloader(self):
    self.train_ds = None

  def _init_valid_dataloader(self):
    self.valid_ds = None


  def _init_test_dataloader(self):
    self.test_ds = None


  def _init_model(self):
    raise NotImplementedError


  def _init_loss_function(self):
    raise NotImplementedError


  def _init_metrics(self):
    raise NotImplementedError


  def setup(self):
    self._init_distribution()


  def _init_distribution(self):
    # Distributed training with TensorFlow
    # Distributed training with Keras
    # check tensorflow mixed precision

    # set all seeds
    tf.keras.utils.set_random_seed(self.hparams.seed)

  def log_basic_info(self, logger):
    logger.info(f'- Tensorflow version: {tf.__version__}')

  def load_trainer_from_checkpoint(self):
    pass

  def setup_checkpoint_saver(self, to_save):
    pass

  def attach_metrics(self, engine, metrics):
    pass

  def train(self, run_params):
    pass

  def validate(self, dl):
    pass

  def evaluate(self, dl):
    pass

  def train_step(self, engine, batch):
    pass

  def loss_backpass(self, loss):
    pass

  def eval_step(self, engine, batch):
    pass

  def test_step(self, engine, batch):
    pass

  def prepare_batch(self, batch, model='valid'):
    pass

  def output_transform(self, x, y, y_pred, loss=None):
    pass

  def _init_scheduler(self):
    self.scheduler = None

  def get_batch(self):
    pass

  def loss_fn(self, y_pred, y):
    raise NotImplementedError

  def _init_optimizer(self):
    raise NotImplementedError

  def _init_criterion_function(self):
    pass

  def _init_logger(self):
    pass

  def clip_gradients(self):
    pass

  def lr_finder(self, min_lr=0.00003, max_lr=10.0, num_iter=None):
    pass


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
