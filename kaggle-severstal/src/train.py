#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Base Training Module'''


# imports
from argparse import ArgumentParser

import pytorch_lightning as pl
from datamodules.classification_datamodule import ClassificationDataModule
from engines.classification_engine import Classifier
from helpers.config_reader import Config, load_config
from helpers.handler_list import HandlerList

#    script imports
# imports


# constants
# constants


# classes
def train_classifier(
  hparams: Config,
  model_name='effnet_b1',
  batch_size=16,
  max_epochs=10,
  ):

  pl.seed_everything(hparams.global_config.seed)

  dm = ClassificationDataModule(
    batch_size=batch_size,
    num_workers=1
  )

  model = Classifier(
    hparams=hparams,
    model_name=model_name,
    lr=0.01,
    weight_decay=1e-4,
    momentum=0.9
  )

  trainer = pl.Trainer(

    accelerator=hparams.global_config.hardware.accelerator,
    strategy=hparams.global_config.hardware.strategy,
    devices=hparams.global_config.hardware.devices,
    num_nodes=hparams.global_config.hardware.num_nodes,
    precision=hparams.global_config.hardware.precision,
    enable_progress_bar=hparams.global_config.hardware.enable_progress_bar,
    deterministic=hparams.global_config.deterministic,


    max_epochs=max_epochs,
    accumulate_grad_batches=hparams.global_config.accumulate_grad_batches,
    logger=HandlerList(hparams.loggers)(),
    callbacks=HandlerList(hparams.callbacks)(),
    log_every_n_steps=20
  )

  trainer.fit(model, dm)

# classes


# functions
def function_name():
  pass
# functions


# main
def main(hparams: Config):
  # train_classifier()

  train_classifier(
    hparams,
    model_name='effnet_b1'
  )


# if main script
if __name__ == '__main__':
  parser = ArgumentParser(parents=[])

  parser.add_argument('--params_yml', type=str)

  arg_params = parser.parse_args()

  hyperparams = load_config(arg_params.params_yml)

  main(hparams=hyperparams)
