#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the metric registry.

The three names are the ones the shipped configs use, and an unknown name
is a clear error, not a `KeyError`.
'''

import pytest

from birdcall.metrics.pl_metric_f1score_clip import SedF1ScoreClip
from birdcall.metrics.pl_metric_f1score_frame import SedF1ScoreFrame
from birdcall.metrics.pl_metric_lraps import SedLRAPS
from birdcall.metrics.pl_registry import METRIC_REGISTRY
from birdcall.metrics.pl_registry import build_metric


def test_build_metric_returns_the_requested_metric():
  '''The registry dispatches on the name.'''
  assert isinstance(build_metric('lraps'), SedLRAPS)
  assert isinstance(build_metric('f1score_clip', threshold=0.5),
                    SedF1ScoreClip)
  assert isinstance(build_metric('f1score_frame', threshold=0.5),
                    SedF1ScoreFrame)
  assert set(METRIC_REGISTRY) == {'lraps', 'f1score_clip', 'f1score_frame'}


def test_build_metric_rejects_an_unknown_name():
  '''An unknown name is a clear error that names the valid ones.'''
  with pytest.raises(ValueError) as error:
    build_metric('not_a_metric')
  message = str(error.value)
  assert 'not_a_metric' in message
  for name in METRIC_REGISTRY:
    assert name in message


def test_the_keyword_arguments_reach_the_metric():
  '''The threshold reaches the metric.'''
  metric = build_metric('f1score_frame', threshold=0.4)
  assert metric.threshold == 0.4
