#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''All model classifier'''


# imports

import pytorch_lightning as pl
import torch
import torch.nn as nn
from metrics.classification_metrics import ClassificationMetrics
from models.classification_models import (EfficientNetB1Classifier,
                                          ResNet34Classifier)

# imports


# constants
# constants


# classes
class Classifier(pl.LightningModule):
  '''All model classifier'''

  def __init__(self, model_name='effnet_b1', lr=0.01, weight_decay=1e-4, momentum=0.9):
    super().__init__()
    self.save_hyperparameters()

    if model_name == 'effnet_b1':
      self.model = EfficientNetB1Classifier()
    elif model_name == 'resnet34':
      self.model = ResNet34Classifier()
    else:
      raise ValueError(f'Unsupported model: {model_name}')

    self.criterion = nn.BCEWithLogitsLoss()

    self.metrics = nn.ModuleDict({
      **ClassificationMetrics()(run_type='train'),
      **ClassificationMetrics()(run_type='val'),
      **ClassificationMetrics()(run_type='test'),
    })

  def metrics_calculation(self, predicted, ground_truth, run_type='train'):
    filter_metrics = nn.ModuleDict({
      k: v for k, v in self.metrics.items() if k.startswith(run_type)
    })

    return {
      f'{metric_name}': metric_fn(predicted, ground_truth)
      for metric_name, metric_fn in filter_metrics.items()
    }

  def train_val_test_steps(self, batch, run_type='train'):
    x, y = batch
    logits = self.model(x)

    loss = self.criterion(logits, y.float())

    preds = torch.sigmoid(logits)

    self.log(f'{run_type}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    for metric_name, calculated in self.metrics_calculation(preds, y, run_type).items():
      self.log(metric_name, calculated, on_step=True, on_epoch=True, prog_bar=True)

    return loss

  def training_step(self, batch, batch_idx):
    return self.train_val_test_steps(batch, run_type='train')

  def validation_step(self, batch, batch_idx):
    return self.train_val_test_steps(batch, run_type='val')

  def test_step(self, batch, batch_idx):
    return self.train_val_test_steps(batch, run_type='test')

    # SGD optimizer
  def configure_optimizers(self):
    optimizer = torch.optim.SGD(
      self.parameters(),
      lr=self.hparams.lr,
      momentum=self.hparams.momentum,
      weight_decay=self.hparams.weight_decay,
      nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode='min',
      patience=2,
      factor=0.5
    )

    return {
      'optimizer': optimizer,
      'lr_scheduler': {
        'scheduler': scheduler,
        'monitor': 'val_loss'
      }
    }

  # for inference
  def predict_proba(self, x):
    logits = self(x)
    return torch.sigmoid(logits)

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
