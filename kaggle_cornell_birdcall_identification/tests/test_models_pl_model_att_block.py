#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the PyTorch attention head.

The softmax is over the **time** axis, so the attention sums to one over
the frames and the clip-level score is the attention-weighted sum of the
per-frame scores.
'''

import pytest
import torch

from birdcall.models.pl_model_att_block import AttBlock

CLASSES_NUM = 264
IN_FEATURES = 1024
FRAMES = 7


def test_attention_sums_to_one_over_time():
  '''The softmax is over the time axis, so the attention sums to one.'''
  block = AttBlock(1024, CLASSES_NUM, activation='sigmoid')
  features = torch.randn(2, 1024, 7)
  clipwise, norm_att, cla = block(features)
  assert clipwise.shape == (2, CLASSES_NUM)
  assert norm_att.shape == (2, CLASSES_NUM, 7)
  assert cla.shape == (2, CLASSES_NUM, 7)
  assert torch.allclose(norm_att.sum(dim=-1), torch.ones(2, CLASSES_NUM),
                         atol=1e-6)
  # The clip-level output is the attention-weighted sum of the frames.
  assert torch.allclose(clipwise, (norm_att * cla).sum(dim=2), atol=1e-6)
  # `cla` is post-sigmoid, so the outputs are probabilities already.
  assert cla.min() >= 0.0 and cla.max() <= 1.0


def test_the_attention_does_not_sum_to_one_over_the_classes():
  '''A softmax over the class axis would sum to one there.'''
  block = AttBlock(IN_FEATURES, CLASSES_NUM, activation='sigmoid')
  features = torch.randn(2, IN_FEATURES, FRAMES)
  _, norm_att, _ = block(features)
  assert not torch.allclose(norm_att.sum(dim=1), torch.ones(2, FRAMES),
                            atol=1e-3)


def test_the_sigmoid_activation_bounds_the_classifier_output():
  '''The sigmoid activation makes the classifier output a probability.'''
  block = AttBlock(IN_FEATURES, CLASSES_NUM, activation='sigmoid')
  features = torch.randn(2, IN_FEATURES, FRAMES)
  _, _, cla = block(features)
  assert cla.min() >= 0.0
  assert cla.max() <= 1.0


def test_the_linear_activation_leaves_the_classifier_alone():
  '''The linear activation is the identity.'''
  block = AttBlock(IN_FEATURES, CLASSES_NUM, activation='linear')
  features = torch.randn(2, IN_FEATURES, FRAMES)
  _, _, cla = block(features)
  assert cla.min() < 0.0


def test_an_unknown_activation_is_refused():
  '''An unsupported activation is a clear error.'''
  block = AttBlock(IN_FEATURES, CLASSES_NUM, activation='softmax')
  with pytest.raises(ValueError, match='Unknown activation'):
    block(torch.randn(2, IN_FEATURES, FRAMES))


def test_the_temperature_is_accepted_and_ignored():
  '''The temperature is dead in the original, and dead here.'''
  first = AttBlock(IN_FEATURES, CLASSES_NUM, temperature=1.0)
  second = AttBlock(IN_FEATURES, CLASSES_NUM, temperature=17.0)
  assert first.temperature == 1.0
  assert second.temperature == 17.0


def test_the_dead_batch_norm_is_kept():
  '''The dead batch norm is kept, because the checkpoint carries it.'''
  block = AttBlock(IN_FEATURES, CLASSES_NUM)
  assert block.bn_att is not None
  assert block.bn_att.num_features == CLASSES_NUM


def test_init_weights_resets_the_batch_norm_to_the_identity():
  '''The batch norm is initialised to weight one and bias zero.'''
  block = AttBlock(IN_FEATURES, CLASSES_NUM)
  block.init_weights()
  assert torch.all(block.bn_att.weight == 1.0)
  assert torch.all(block.bn_att.bias == 0.0)
