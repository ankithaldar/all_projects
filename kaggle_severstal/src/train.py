#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports

import pytorch_lightning as pl
from callbacks.get_callbacks import Callbacks
from datamodules.classification_datamodule import ClassificationDataModule
from engines.classification_engine import Classifier
from loggers.get_loggers import Loggers

#    script imports
# imports


# constants
# constants


# classes
def train_classifier(
  model_name='effnet_b1',
  batch_size=16,
  max_epochs=10,
):

  pl.seed_everything(42)

  dm = ClassificationDataModule(
    batch_size=batch_size,
    num_workers=1
  )

  model = Classifier(
    model_name=model_name,
    lr=0.01,
    weight_decay=1e-4,
    momentum=0.9
  )

  trainer = pl.Trainer(
    accelerator='cuda',
    devices=1,
    precision='bf16-mixed',
    max_epochs=max_epochs,
    accumulate_grad_batches=4,
    logger=Loggers()(),
    callbacks=Callbacks()(),
    log_every_n_steps=20
  )

  trainer.fit(model, dm)

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
  train_classifier()
  train_classifier(model_name='resnet34')
