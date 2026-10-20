#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the PyTorch model registry.

Only the DenseNet-121 variant is shipped: it is the model all thirteen
ensemble members use.  An unknown name is a clear error, not an
`AttributeError`.
'''

import pytest

from birdcall.models.pl_model_panns_dense121_att import PANNsDense121Att
from birdcall.models.pl_registry import MODEL_REGISTRY
from birdcall.models.pl_registry import build_model

SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
CLASSES_NUM = 264


def test_build_model_raises_for_an_unknown_name():
  '''An unknown name names the valid keys.'''
  with pytest.raises(ValueError) as error:
    build_model('sed_nope')
  assert 'sed_nope' in str(error.value)
  assert 'sed_dense121att' in str(error.value)


def test_build_model_builds_the_registered_model():
  '''The registry builds the DenseNet-121 variant.'''
  assert MODEL_REGISTRY['sed_dense121att'] is PANNsDense121Att
  model = build_model('sed_dense121att', sample_rate=SAMPLE_RATE,
                      window_size=WINDOW_SIZE, hop_size=HOP_SIZE,
                      mel_bins=MEL_BINS, fmin=50, fmax=14000,
                      classes_num=CLASSES_NUM, apply_aug=True, top_db=None)
  assert isinstance(model, PANNsDense121Att)


def test_the_registry_holds_the_one_shipped_model():
  '''The other four PANNs variants have no shipped config.'''
  assert list(MODEL_REGISTRY) == ['sed_dense121att']


def test_the_keyword_arguments_reach_the_model():
  '''The constructor arguments are the ones of the model.'''
  model = build_model('sed_dense121att', sample_rate=SAMPLE_RATE,
                      window_size=WINDOW_SIZE, hop_size=HOP_SIZE,
                      mel_bins=MEL_BINS, fmin=50, fmax=14000,
                      classes_num=CLASSES_NUM, apply_aug=True, top_db=None)
  assert model.sample_rate == SAMPLE_RATE
  assert model.window_size == WINDOW_SIZE
  assert model.hop_size == HOP_SIZE
  assert model.mel_bins == MEL_BINS
  assert model.classes_num == CLASSES_NUM
  assert model.apply_aug is True
