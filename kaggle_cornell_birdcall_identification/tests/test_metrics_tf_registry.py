#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the Keras metric registry.

The three names are the ones the shipped configs use, and an unknown name
is a clear error, not a `KeyError`.
'''

import pytest

from birdcall.metrics.tf_registry import METRIC_REGISTRY
from birdcall.metrics.tf_registry import build_metric


def test_the_metric_registry_holds_the_three_metrics() -> None:
  '''The three validation metrics are registered.'''
  assert sorted(METRIC_REGISTRY) == ['f1score_clip', 'f1score_frame', 'lraps']


def test_the_metrics_are_built_with_their_parameters() -> None:
  '''The threshold reaches the metric.'''
  metric = build_metric('f1score_frame', threshold=0.4)
  assert metric.threshold == 0.4
  assert metric.name == 'f1score_frame'


def test_an_unknown_metric_raises() -> None:
  '''An unknown metric name is a clear error.'''
  with pytest.raises(ValueError, match='Unknown metric'):
    build_metric('auc')
