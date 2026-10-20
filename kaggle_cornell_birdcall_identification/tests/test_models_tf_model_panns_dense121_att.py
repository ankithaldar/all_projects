#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the Keras PANNs DenseNet-121 model.

The four things this model gets wrong easily are asserted here: the
ImageNet preprocessing bypass, the measured DenseNet time axis, the
tail padding of the frame head, and the mixup halving.
'''


import numpy as np
import pytest
import tensorflow as tf

from birdcall.config_params.model_config import ModelConfig
from birdcall.models.tf_model_panns_dense121_att import PANNsDense121Att


SR = 32000
WINDOW = 1024
HOP = 320
MELS = 64
FMIN = 50
FMAX = 14000
CLASSES = 264
THIRTY_SECONDS = 32000 * 30

def build_model(apply_aug: bool = True) -> PANNsDense121Att:
  '''Builds the model of the shipped configuration.

  Args:
    apply_aug: Whether the in-graph SpecAugment runs.

  Returns:
    The model.
  '''
  return PANNsDense121Att(sample_rate=SR, window_size=WINDOW, hop_size=HOP,
                           mel_bins=MELS, fmin=FMIN, fmax=FMAX,
                           classes_num=CLASSES, apply_aug=apply_aug,
                           top_db=None)

def waveform(samples: int, seed: int = 3) -> np.ndarray:
  '''Builds a deterministic waveform batch.

  Args:
    samples: The number of samples of every row.
    seed: The seed of the generator.

  Returns:
    A ``(batch, snippets, samples)`` float32 tensor.
  '''
  rng = np.random.default_rng(seed)
  return rng.standard_normal((2, 1, samples)).astype(np.float32)

def test_the_output_is_the_two_heads() -> None:
  '''The model returns the clip and the frame heads.'''
  model = build_model()
  out = model(waveform(32000), training=False)
  assert set(out) == {'clipwise_output', 'framewise_output'}
  assert out['clipwise_output'].shape == (2, 1, CLASSES)
  assert out['framewise_output'].shape[0] == 2
  assert out['framewise_output'].shape[2] == 101

def test_the_thirty_second_shapes() -> None:
  '''A 30 second waveform gives 3001 frames and 264 classes.'''
  model = build_model()
  out = model(waveform(THIRTY_SECONDS), training=False)
  assert out['clipwise_output'].shape == (2, 1, CLASSES)
  assert out['framewise_output'].shape == (2, 1, 3001, CLASSES)

def test_the_densenet_time_axis_is_93() -> None:
  '''The DenseNet time axis is measured, not assumed.'''
  model = build_model()
  flat = waveform(THIRTY_SECONDS)[0, 0]
  logmel = model.frontend(flat[None, :])
  assert logmel.shape == (1, 3001, MELS)
  widened = tf.broadcast_to(tf.expand_dims(logmel, -1), (1, 3001, MELS, 3))
  features = model.backbone(widened)
  assert features.shape[1] == 93
  assert features.shape[3] == 1024

def test_the_tail_pad_is_exactly_25_frames() -> None:
  '''The frame head is padded with 25 repeats of the last frame.'''
  model = build_model()
  out = model(waveform(THIRTY_SECONDS), training=False)
  frames = out['framewise_output'].numpy()[0, 0]
  assert frames.shape == (3001, CLASSES)
  # 93 backbone frames times 32 is 2976, so 25 frames are padded.
  assert 3001 - 93 * 32 == 25
  tail = frames[2976:]
  assert tail.shape == (25, CLASSES)
  assert np.allclose(tail, tail[0], atol=1e-6)
  # The padded frames are a copy of the last interpolated frame, and
  # not a block of "no bird".
  assert not np.allclose(frames[2900], tail[0], atol=1e-6)

def test_the_imagenet_preprocessing_is_bypassed() -> None:
  '''The first convolution sees the raw bn0 normalised log-mel.'''
  model = build_model()
  audio = waveform(THIRTY_SECONDS)[0, 0]
  model(audio[None, None, :], training=False)
  seen = model.last_backbone_input.numpy()
  assert seen.shape == (1, 3001, MELS, 3)
  # The three channels are the same log-mel: a broadcast, not a copy.
  assert np.allclose(seen[:, :, :, 0], seen[:, :, :, 2], atol=1e-6)
  expected = model.bn0(model.frontend(audio[None, :]), training=False)
  assert np.allclose(seen[:, :, :, 0], expected.numpy(), atol=1e-5)
  # The ImageNet scaling is emphatically not what the backbone sees.
  # (The copy matters: `preprocess_input` works in place.)
  scaled = tf.keras.applications.densenet.preprocess_input(seen.copy())
  assert not np.allclose(seen, scaled, atol=1e-3)
  assert model.imagenet_preprocessing_bypassed is True

def test_the_first_backbone_layer_does_not_scale() -> None:
  '''The backbone starts with a padding layer, not a rescaling one.'''
  model = build_model()
  first = model.backbone.layers[1]
  assert isinstance(first, tf.keras.layers.ZeroPadding2D)

def test_the_batch_norm_comes_before_the_spec_augment() -> None:
  '''The ordering of bn0 and SpecAugment is load bearing.'''
  model = build_model()
  order = []
  original_bn0 = model.bn0.call
  original_aug = model.spec_augmenter.call

  def spy_bn0(inputs, training=False, **kwargs):
    order.append('bn0')
    return original_bn0(inputs, training=training, **kwargs)

  def spy_aug(inputs, training=False, **kwargs):
    order.append('spec_augment')
    return original_aug(inputs, training=training, **kwargs)

  model.bn0.call = spy_bn0
  model.spec_augmenter.call = spy_aug
  model(waveform(32000), training=True)
  assert order == ['bn0', 'spec_augment']

def test_the_spec_augment_is_off_at_inference() -> None:
  '''The SpecAugment must not run when the model is not training.'''
  model = build_model()
  audio = waveform(32000)
  model(audio, training=False)
  first = model.last_backbone_input.numpy()
  model(audio, training=False)
  second = model.last_backbone_input.numpy()
  assert np.allclose(first, second, atol=1e-6)

def test_the_mixup_halves_the_batch() -> None:
  '''The mixup halves the batch, and the labels with it.'''
  model = build_model()
  audio = waveform(32000)
  mixup_lambda = tf.constant([0.25, 0.75], dtype=tf.float32)
  out = model((audio, mixup_lambda), training=True)
  assert out['clipwise_output'].shape == (1, 1, CLASSES)
  assert out['framewise_output'].shape[0] == 1

def test_the_mix_labels_halves_and_mixes() -> None:
  '''The labels are mixed with the same coefficients and halved.'''
  model = build_model()
  labels = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
  mixup_lambda = tf.constant([0.25, 0.75], dtype=tf.float32)
  mixed = model.mix_labels(tf.constant(labels), mixup_lambda).numpy()
  assert mixed.shape == (1, 1, 2)
  assert np.allclose(mixed, [[0.25, 0.75]])

def test_mixup_without_augmentation_does_not_halve_the_batch() -> None:
  '''The halving is guarded by the augmentation flag.'''
  model = build_model(apply_aug=False)
  audio = waveform(32000)
  mixup_lambda = tf.constant([0.25, 0.75], dtype=tf.float32)
  out = model((audio, mixup_lambda), training=True)
  assert out['clipwise_output'].shape == (2, 1, CLASSES)

def test_the_model_accepts_a_bare_waveform() -> None:
  '''The model can be called without the mixup coefficients.'''
  model = build_model()
  out = model(waveform(32000), training=True)
  assert out['clipwise_output'].shape == (2, 1, CLASSES)

def test_the_config_round_trips() -> None:
  '''The configuration of the model can be read back.'''
  model = build_model()
  config = model.get_config()
  assert config['window_size'] == WINDOW
  assert config['hop_size'] == HOP
  assert config['mel_bins'] == MELS
  assert config['classes_num'] == CLASSES
  assert config['apply_aug'] is True
  assert config['top_db'] is None

def test_the_model_is_built_from_the_shipped_configuration() -> None:
  '''The model takes the configuration of the config layer.'''
  config = ModelConfig()
  model = PANNsDense121Att(**vars(config))
  assert model.mel_bins == config.mel_bins
  assert model.classes_num == config.classes_num
  assert model.interpolate_ratio == 32
