#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Create lists for each of loggers, callbacks, metrics etc'''


# imports

from helpers.module_importer import instantiate_class

#    script imports
# imports


# constants
# constants


# classes
class HandlerList:
  '''Create lists for each of loggers, callbacks, metrics etc'''

  def __init__(self, handler_list):
    self.handler_list = handler_list

  def __call__(self):
    handler_lists = []
    if isinstance(self.handler_list, list) and len(self.handler_list) > 0:
      handler_lists = [
        instantiate_class(handler_lists.class_path, handler_lists.init_args)
        for handler_lists in self.handler_list
      ]

    return handler_lists


class MetricsHandler:
  '''Metrics for classification'''

  def __init__(self, metrics_list):
    self.metrics_list = metrics_list

  def __call__(self, run_type='train'):
    metrics = {}
    if isinstance(self.metrics_list, list) and len(self.metrics_list) > 0:
      metrics = {
        f'{run_type}_{metric.class_path.strip().rsplit(".", maxsplit=1)[-1]}': instantiate_class(metric.class_path, metric.init_args)
        for metric in self.metrics_list
      }

    return metrics

# classes
