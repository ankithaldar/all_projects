#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Classification Metrics'''


# imports

from dataclasses import dataclass

from helpers.module_importer import instantiate_class

#    script imports
# imports


# constants
# constants


# classes
@dataclass
class Metrics:
  class_path: str
  init_args: dict


def get_metrics():
  return [
    Metrics(
      class_path='torchmetrics.Accuracy',
      init_args={'task': 'binary', 'threshold': 0.5}
    ),
    Metrics(
      class_path='torchmetrics.F1Score',
      init_args={'task': 'binary', 'threshold': 0.5}
    ),
    Metrics(
      class_path='torchmetrics.Precision',
      init_args={'task': 'binary', 'threshold': 0.5}
    ),
    Metrics(
      class_path='torchmetrics.Recall',
      init_args={'task': 'binary', 'threshold': 0.5}
    ),
  ]


class ClassificationMetrics:
  '''Metrics for classification'''

  def __init__(self):
    self.metrics_list = get_metrics()

  def __call__(self, run_type='train'):
    metrics = {}
    if isinstance(self.metrics_list, list) and len(self.metrics_list) > 0:
      metrics = {
        f'{run_type}_{metric.class_path.strip().rsplit(".", maxsplit=1)[-1]}': instantiate_class(metric.class_path, metric.init_args)
        for metric in self.metrics_list
      }

    return metrics

# classes
