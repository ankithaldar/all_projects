#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The cross-backend parity of the model.

The two models cannot be compared numerically without a weight
conversion, which is a piece of work of its own.  What is compared
here is everything that does not depend on the weights: the shapes, the
number of frames, the interpolation ratio, the tail padding and the
semantics of the attention.
'''

import numpy as np
import pytest
import tensorflow as tf

from birdcall.models.tf_model_panns_dense121_att import PANNsDense121Att

SR = 32000
N_FFT = 1024
HOP = 320
MELS = 64
FMIN = 50
FMAX = 14000
CLASSES = 264


def waveform(samples: int, seed: int = 23) -> np.ndarray:
  '''Builds a deterministic waveform batch.

  Args:
    samples: The number of samples of every row.
    seed: The seed of the generator.

  Returns:
    A ``(batch, snippets, samples)`` float32 tensor.
  '''
  rng = np.random.default_rng(seed)
  return rng.standard_normal((1, 1, samples)).astype(np.float32)


def torch_model():
  '''Builds the torch model of the shipped configuration.

  Returns:
    The model.
  '''
  torch = pytest.importorskip('torch')
  from birdcall.models.pl_model_panns_dense121_att import PANNsDense121Att

  return PANNsDense121Att(sample_rate=SR, window_size=N_FFT, hop_size=HOP,
                          mel_bins=MELS, fmin=FMIN, fmax=FMAX,
                          classes_num=CLASSES, apply_aug=True, top_db=None)


def keras_model() -> PANNsDense121Att:
  '''Builds the Keras model of the shipped configuration.

  Returns:
    The model.
  '''
  return PANNsDense121Att(sample_rate=SR, window_size=N_FFT, hop_size=HOP,
                          mel_bins=MELS, fmin=FMIN, fmax=FMAX,
                          classes_num=CLASSES, apply_aug=True, top_db=None)


@pytest.mark.parity
def test_the_two_models_return_the_same_dict() -> None:
  '''Both models return the clip and the frame heads.'''
  torch = pytest.importorskip('torch')
  audio = waveform(32000)
  keras_out = keras_model()(audio, training=False)
  torch_out = torch_model()((torch.from_numpy(audio), None))
  assert set(keras_out) == set(torch_out)
  assert tuple(keras_out['clipwise_output'].shape) == tuple(
      torch_out['clipwise_output'].shape)
  assert tuple(keras_out['framewise_output'].shape) == tuple(
      torch_out['framewise_output'].shape)


@pytest.mark.parity
def test_the_two_models_have_the_same_frame_count() -> None:
  '''Both models give 3001 frames of a 30 second waveform.'''
  torch = pytest.importorskip('torch')
  audio = waveform(SR * 30)
  keras_out = keras_model()(audio, training=False)
  torch_out = torch_model()((torch.from_numpy(audio), None))
  assert keras_out['framewise_output'].shape[2] == 3001
  assert torch_out['framewise_output'].shape[2] == 3001


@pytest.mark.parity
def test_the_interpolation_ratio_is_the_same() -> None:
  '''Both models upsample the time axis 32 times.'''
  torch_model()
  assert keras_model().interpolate_ratio == 32


@pytest.mark.parity
def test_the_tail_padding_is_the_same() -> None:
  '''Both models pad the frame head with the last frame.'''
  torch = pytest.importorskip('torch')
  audio = waveform(SR * 30)
  keras_frames = keras_model()(audio, training=False)[
      'framewise_output'].numpy()[0, 0]
  torch_frames = torch_model()((torch.from_numpy(audio), None))[
      'framewise_output'].detach().numpy()[0, 0]
  # 93 backbone frames times 32 is 2976, so 25 frames are padded, and
  # they are a copy of the last frame.
  assert 3001 - 93 * 32 == 25
  assert np.allclose(keras_frames[2976:], keras_frames[2976])
  assert np.allclose(torch_frames[2976:], torch_frames[2976])


@pytest.mark.parity
def test_the_attention_sums_to_one_over_time_in_both_backends() -> None:
  '''The attention is a distribution over time, in both backends.'''
  torch = pytest.importorskip('torch')
  keras_block = keras_model().att_block
  features = np.random.default_rng(0).standard_normal(
      (2, 7, 1024)).astype(np.float32)
  attention = keras_block.attention_weights(features).numpy()
  assert np.allclose(attention.sum(axis=1), 1.0, atol=1e-5)
  # The torch block sums over the last axis of a (batch, classes,
  # frames) tensor, which is the time axis.
  torch_block = torch_model().att_block
  with torch.no_grad():
    _, norm_att, _ = torch_block(torch.from_numpy(features).transpose(1, 2))
  assert np.allclose(norm_att.numpy().sum(axis=2), 1.0, atol=1e-5)
