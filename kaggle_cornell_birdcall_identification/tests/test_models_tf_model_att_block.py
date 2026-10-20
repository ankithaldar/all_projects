#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the Keras attention block.

The softmax axis is the single highest probability silent failure of
the whole port, so it is asserted from both sides: the attention sums
to one over the time axis, and it does **not** sum to one over the
class axis.
'''

import numpy as np
import pytest

from birdcall.models.tf_model_att_block import ATT_AXIS
from birdcall.models.tf_model_att_block import AttBlock

IN_FEATURES = 16
OUT_FEATURES = 7
FRAMES = 11


def features(seed: int = 5) -> np.ndarray:
  '''Builds a deterministic feature tensor.

  Args:
    seed: The seed of the generator.

  Returns:
    A ``(2, frames, in features)`` float32 tensor.
  '''
  rng = np.random.default_rng(seed)
  return rng.standard_normal((2, FRAMES, IN_FEATURES)).astype(np.float32)


def test_the_output_holds_the_three_readings() -> None:
  '''The block returns the clip score, the attention and the classifier.'''
  block = AttBlock(IN_FEATURES, OUT_FEATURES, activation='sigmoid')
  out = block(features())
  assert set(out) == {'clipwise_output', 'norm_att', 'cla'}
  assert out['clipwise_output'].shape == (2, OUT_FEATURES)
  assert out['norm_att'].shape == (2, FRAMES, OUT_FEATURES)
  assert out['cla'].shape == (2, FRAMES, OUT_FEATURES)


def test_the_attention_sums_to_one_over_time() -> None:
  '''The softmax is over the time axis.'''
  block = AttBlock(IN_FEATURES, OUT_FEATURES, activation='sigmoid')
  attention = block(features())['norm_att'].numpy()
  assert np.allclose(attention.sum(axis=1), 1.0, atol=1e-5)
  assert np.all(attention >= 0.0)


def test_the_attention_does_not_sum_to_one_over_the_classes() -> None:
  '''A softmax over the class axis would sum to one there.'''
  block = AttBlock(IN_FEATURES, OUT_FEATURES, activation='sigmoid')
  attention = block(features())['norm_att'].numpy()
  assert not np.allclose(attention.sum(axis=2), 1.0, atol=1e-3)


def test_the_softmax_axis_is_minus_two() -> None:
  '''The axis is written down, because -1 is a silent wrong answer.'''
  assert ATT_AXIS == -2
  block = AttBlock(IN_FEATURES, OUT_FEATURES)
  assert block.softmax.axis == -2


def test_the_clipwise_output_is_the_sum_over_time() -> None:
  '''The clip score is the attention weighted sum over the frames.'''
  block = AttBlock(IN_FEATURES, OUT_FEATURES, activation='sigmoid')
  out = block(features())
  expected = (out['norm_att'] * out['cla']).numpy().sum(axis=1)
  assert np.allclose(out['clipwise_output'].numpy(), expected, atol=1e-5)


def test_the_sigmoid_is_applied_to_the_classifier() -> None:
  '''The classifier output is a probability.'''
  block = AttBlock(IN_FEATURES, OUT_FEATURES, activation='sigmoid')
  cla = block(features())['cla'].numpy()
  assert cla.min() >= 0.0
  assert cla.max() <= 1.0


def test_the_linear_activation_leaves_the_classifier_alone() -> None:
  '''The linear activation is the identity.'''
  block = AttBlock(IN_FEATURES, OUT_FEATURES, activation='linear')
  out = block(features())
  assert out['cla'].numpy().min() < 0.0


def test_an_unknown_activation_is_refused() -> None:
  '''An unsupported activation raises.'''
  with pytest.raises(ValueError, match='Unknown activation'):
    AttBlock(IN_FEATURES, OUT_FEATURES, activation='softmax')


def test_the_temperature_is_dead() -> None:
  '''The temperature is accepted and ignored, as in the original.'''
  first = AttBlock(IN_FEATURES, OUT_FEATURES, temperature=1.0)
  second = AttBlock(IN_FEATURES, OUT_FEATURES, temperature=17.0)
  assert first.temperature == 1.0
  assert second.temperature == 17.0


def test_the_dead_batch_norm_is_kept() -> None:
  '''The dead batch norm is kept, because the checkpoint carries it.'''
  block = AttBlock(IN_FEATURES, OUT_FEATURES)
  assert block.bn_att is not None
  assert block.bn_att.axis == -1


def test_the_attention_is_inspectable() -> None:
  '''The last attention is exposed for inspection.'''
  block = AttBlock(IN_FEATURES, OUT_FEATURES, activation='sigmoid')
  block(features())
  assert block.last_attention is not None
  assert np.allclose(block.last_attention.numpy().sum(axis=1), 1.0, atol=1e-5)


def test_the_attention_weights_can_be_computed_on_their_own() -> None:
  '''The attention of a feature tensor is available without the sum.'''
  block = AttBlock(IN_FEATURES, OUT_FEATURES, activation='sigmoid')
  x = features()
  attention = block.attention_weights(x).numpy()
  assert attention.shape == (2, FRAMES, OUT_FEATURES)
  assert np.allclose(attention.sum(axis=1), 1.0, atol=1e-5)


def test_the_config_round_trips() -> None:
  '''The configuration of the block can be read back.'''
  block = AttBlock(IN_FEATURES, OUT_FEATURES, activation='sigmoid',
                   temperature=2.0)
  config = block.get_config()
  assert config['in_features'] == IN_FEATURES
  assert config['out_features'] == OUT_FEATURES
  assert config['activation'] == 'sigmoid'
  assert config['temperature'] == 2.0
