#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the Keras training wrapper.

The wrapper is where the mixup halving happens, because the batch axis
changes its meaning in the middle of the forward pass and the labels
must be halved with it.
'''

import numpy as np
import pytest
import tensorflow as tf

from birdcall.engines.tf_keras_model import SedKerasModel
from birdcall.losses.tf_loss_scaled_pos_neg_focal import (
    SedScaledPosNegFocalLoss)

CLASSES = 6
BATCH = 2
SNIPPETS = 2
SAMPLES = 64


class TinyBackbone(tf.keras.Model):
  '''A backbone that mimics the real one without the DenseNet.'''

  def __init__(self) -> None:
    super().__init__()
    self.apply_aug = True
    self.dense = tf.keras.layers.Dense(CLASSES, activation='sigmoid')
    self.frames = tf.keras.layers.Dense(CLASSES, activation='sigmoid')

  def call(self, inputs, training: bool = False):
    '''Runs the fake backbone.

    Args:
      inputs: The waveform, or the tuple with the mixup coefficients.
      training: Whether the model is training.

    Returns:
      A dict of the two heads.
    '''
    del training
    waveform, mixup_lambda = (inputs if isinstance(inputs, (tuple, list))
                              else (inputs, None))
    shape = tf.shape(waveform)
    flat = tf.reshape(waveform, (shape[0] * shape[1], shape[2]))
    batch = shape[0]
    snippets = shape[1]
    if mixup_lambda is not None:
      even = flat[0::2]
      odd = flat[1::2]
      lam = tf.reshape(mixup_lambda[0::2], (-1, 1))
      flat = even * lam + odd * (1.0 - lam)
      batch = (shape[0] * shape[1]) // 2
      snippets = tf.constant(1, tf.int32)
    features = self.dense(flat)
    frames = tf.repeat(features[:, None, :], 4, axis=1)
    return {
        'clipwise_output': tf.reshape(features, (batch, snippets, CLASSES)),
        'framewise_output': tf.reshape(frames, (batch, snippets, 4,
                                                 CLASSES)),
    }

  def mix_labels(self, labels, mixup_lambda):
    '''Mixes and halves a label tensor.

    Args:
      labels: The label tensor.
      mixup_lambda: The mixup coefficients.

    Returns:
      The mixed and halved label tensor.
    '''
    shape = tf.shape(labels)
    flat = tf.reshape(labels, (shape[0] * shape[1], shape[2]))
    even = flat[0::2]
    odd = flat[1::2]
    lam = tf.reshape(mixup_lambda[0::2], (-1, 1))
    mixed = even * lam + odd * (1.0 - lam)
    return tf.reshape(mixed, ((shape[0] * shape[1]) // 2, 1, shape[2]))


def labels() -> np.ndarray:
  '''Builds a deterministic label batch.

  Returns:
    A ``(batch, snippets, classes)`` float32 array.
  '''
  rng = np.random.default_rng(0)
  return (rng.random((BATCH, SNIPPETS, CLASSES)) > 0.7).astype('float32')


def waveforms() -> np.ndarray:
  '''Builds a deterministic waveform batch.

  Returns:
    A ``(batch, snippets, samples)`` float32 array.
  '''
  rng = np.random.default_rng(1)
  return rng.random((BATCH, SNIPPETS, SAMPLES)).astype('float32')


def build_model(apply_mixup: bool = False) -> SedKerasModel:
  '''Builds the training wrapper of the fake backbone.

  Args:
    apply_mixup: Whether the mixup runs.

  Returns:
    The model.
  '''
  criterion = SedScaledPosNegFocalLoss(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                                       secondary_factor=1.0)
  return SedKerasModel(TinyBackbone(), criterion=criterion,
                       apply_mixup=apply_mixup)


def test_the_loss_metrics_follow_the_loss_keys() -> None:
  '''One running mean per loss key.'''
  model = build_model()
  assert sorted(model.loss_metrics) == sorted(model.criterion.loss_keys)


def test_the_train_step_returns_the_loss() -> None:
  '''The training step returns the loss of the batch.'''
  model = build_model()
  model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                loss=model.criterion)
  out = model.train_step(({'waveforms': waveforms()},
                           {'all_labels': labels(),
                            'secondary_labels': labels() * 0.0}))
  # Only scalars are returned: Keras aggregates the value of a training
  # step over the batch.
  assert set(out) == {'loss'}
  assert float(out['loss']) > 0.0
  # The loss components are available for the loggers.
  assert model.last_components
  assert 'bce_loss' in model.last_components


def test_the_mixup_halves_the_labels() -> None:
  '''The labels are halved with the batch.'''
  model = build_model(apply_mixup=True)
  model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                loss=model.criterion)
  out = model.train_step(({'waveforms': waveforms()},
                           {'all_labels': labels(),
                            'secondary_labels': labels() * 0.0}))
  assert float(out['loss']) > 0.0
  # The batch of two files of two snippets is four rows, halved to two.
  assert tuple(model.last_labels['all_labels'].shape) == (2, 1, CLASSES)


def test_the_mixup_coefficients_sum_to_one() -> None:
  '''The coefficients of a pair sum to one, as in the original.'''
  model = build_model(apply_mixup=True)
  # pylint: disable=protected-access
  first = model._draw_mixup_lambda(4).numpy()
  second = model._draw_mixup_lambda(4).numpy()
  assert first.shape == (4,)
  assert np.allclose(first[0::2] + first[1::2], 1.0)
  # The sequence advances, like the original's RandomState.
  assert not np.allclose(first, second)


def test_the_mixup_sequence_is_reproducible() -> None:
  '''Two models with the same seed draw the same coefficients.'''
  first = build_model(apply_mixup=True)
  second = build_model(apply_mixup=True)
  # pylint: disable=protected-access
  assert np.allclose(first._draw_mixup_lambda(4).numpy(),
                    second._draw_mixup_lambda(4).numpy())


def test_the_test_step_returns_the_torch_dict() -> None:
  '''The evaluation step returns the dict the torch engine returns.'''
  model = build_model()
  model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                loss=model.criterion)
  out = model.test_step(({'waveforms': waveforms()},
                          {'all_labels': labels(),
                           'secondary_labels': labels() * 0.0}))
  assert set(out) == {'loss', 'x', 'y_pred', 'y', 'dict_loss'}
  assert tuple(out['y_pred']['clipwise_output'].shape) == (BATCH, SNIPPETS,
                                                            CLASSES)


def test_the_compute_loss_uses_the_criterion() -> None:
  '''The argument order of the criterion is handled by the override.'''
  model = build_model()
  y_pred = {'clipwise_output': np.full((BATCH, SNIPPETS, CLASSES), 0.5,
                                       dtype='float32')}
  value = model.compute_loss(x=None, y={'all_labels': labels()},
                              y_pred=y_pred)
  assert float(value) > 0.0


def test_the_loss_metrics_are_updated() -> None:
  '''The running means of the loss keys see the batch.'''
  model = build_model()
  model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                loss=model.criterion)
  model.train_step(({'waveforms': waveforms()},
                    {'all_labels': labels(),
                     'secondary_labels': labels() * 0.0}))
  assert float(model.loss_metrics['bce_loss'].result()) > 0.0
  model.reset_loss_metrics()
  assert float(model.loss_metrics['bce_loss'].result()) == 0.0


def test_a_model_without_a_criterion_refuses_to_compute_a_loss() -> None:
  '''A model without a criterion says so.'''
  model = SedKerasModel(TinyBackbone())
  with pytest.raises(ValueError, match='criterion'):
    model.compute_loss(x=None, y=None, y_pred=None)


def test_the_model_survives_a_save_and_load_round_trip(tmp_path) -> None:
  '''A checkpoint reloads, with the same weights.'''
  from birdcall.models.tf_registry import build_model as build_backbone

  backbone = build_backbone('sed_dense121att', sample_rate=32000,
                             window_size=1024, hop_size=320, mel_bins=64,
                             fmin=50, fmax=14000, classes_num=8,
                             apply_aug=True, top_db=None)
  criterion = SedScaledPosNegFocalLoss(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                                       secondary_factor=1.0)
  model = SedKerasModel(backbone, criterion=criterion)
  audio = np.random.default_rng(3).standard_normal(
      (1, 1, 32000)).astype('float32')
  before = model.backbone(audio, training=False)['clipwise_output'].numpy()
  path = tmp_path / 'model.keras'
  model.save(path)
  restored = tf.keras.models.load_model(path)
  after = restored.backbone(audio, training=False)['clipwise_output'].numpy()
  assert np.max(np.abs(before - after)) == 0.0


def test_the_real_backbone_trains() -> None:
  '''One training step of the real model.'''
  from birdcall.models.tf_registry import build_model as build_backbone

  backbone = build_backbone('sed_dense121att', sample_rate=32000,
                             window_size=1024, hop_size=320, mel_bins=64,
                             fmin=50, fmax=14000, classes_num=CLASSES,
                             apply_aug=True, top_db=None)
  criterion = SedScaledPosNegFocalLoss(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                                       secondary_factor=1.0)
  model = SedKerasModel(backbone, criterion=criterion)
  model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss=criterion)
  rng = np.random.default_rng(2)
  audio = rng.standard_normal((1, 1, 32000)).astype('float32')
  target = np.zeros((1, 1, CLASSES), dtype='float32')
  out = model.train_step(({'waveforms': audio},
                           {'all_labels': target,
                            'secondary_labels': target}))
  assert np.isfinite(float(out['loss']))
