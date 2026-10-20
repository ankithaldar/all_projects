#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the PyTorch PANNs DenseNet-121 attention model.

The model is exercised on a two second synthetic waveform, so the whole
suite runs in seconds on a CPU.  The frame arithmetic is the interesting
part: the backbone downsamples the time axis by x32, and the tail of the
framewise output is the last frame, repeated.
'''

import pytest
import torch

from birdcall.models.pl_model_panns_dense121_att import PANNsDense121Att
from birdcall.models.pl_model_panns_dense121_att import interpolate
from birdcall.models.pl_model_panns_dense121_att import pad_framewise_output

SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
CLASSES_NUM = 264
N_SAMPLES = 2 * SAMPLE_RATE
FRAMES = 1 + N_SAMPLES // HOP_SIZE
# The frame count of a thirty second clip at 32 kHz.
FRAMES_30S = 1 + 30 * SAMPLE_RATE // HOP_SIZE


def _build_model(apply_aug: bool = True) -> PANNsDense121Att:
  '''Builds the model of the shipped configuration, in eval mode.'''
  torch.manual_seed(0)
  model = PANNsDense121Att(SAMPLE_RATE, WINDOW_SIZE, HOP_SIZE, MEL_BINS,
                           50, 14000, CLASSES_NUM, apply_aug, None)
  model.eval()
  return model


@pytest.fixture(scope='module')
def model() -> PANNsDense121Att:
  '''The model under test.'''
  return _build_model()


def _densenet_time_axis(frames: int) -> int:
  '''The time axis of the DenseNet-121 features, x32 with floors.

  Args:
    frames: The number of log-mel frames.

  Returns:
    The number of backbone frames.
  '''
  # conv0: 7x7, stride 2, padding 3.
  frames = (frames + 2 * 3 - 7) // 2 + 1
  # pool0: 3x3, stride 2, padding 1.
  frames = (frames + 2 * 1 - 3) // 2 + 1
  # transition1, transition2, transition3: 2x2 average pooling.
  for _ in range(3):
    frames //= 2
  return frames


def test_model_output_shapes(model):
  '''The two outputs are (B, C, 264) and (B, C, n_frames, 264).'''
  waveform = torch.randn(2, 1, N_SAMPLES)
  out = model((waveform, None))
  assert out['clipwise_output'].shape == (2, 1, CLASSES_NUM)
  assert out['framewise_output'].shape == (2, 1, FRAMES, CLASSES_NUM)


def test_model_frame_count_is_one_plus_samples_over_hop(model):
  '''The frame count is the centred log-mel frame count.'''
  waveform = torch.randn(1, 1, N_SAMPLES)
  out = model((waveform, None))
  assert out['framewise_output'].shape[2] == 1 + N_SAMPLES // HOP_SIZE


def test_tail_pad_repeats_the_last_frame(model):
  '''The tail of the framewise output is the last frame, repeated.'''
  waveform = torch.randn(1, 1, N_SAMPLES)
  out = model((waveform, None))
  framewise = out['framewise_output'][0, 0]
  backbone_frames = _densenet_time_axis(FRAMES)
  pad = FRAMES - backbone_frames * 32
  assert backbone_frames * 32 < FRAMES
  assert pad > 0
  assert torch.allclose(framewise[-pad:], framewise[-1:].expand(pad, -1))


def test_thirty_second_tail_pad_is_25_frames():
  '''A thirty second clip is 3001 frames, of which 25 are tail repeat.'''
  backbone_frames = _densenet_time_axis(FRAMES_30S)
  assert FRAMES_30S == 3001
  assert backbone_frames == 93
  assert backbone_frames * 32 == 2976
  framewise = torch.randn(1, 2976, CLASSES_NUM)
  padded = pad_framewise_output(framewise, FRAMES_30S)
  assert padded.shape == (1, FRAMES_30S, CLASSES_NUM)
  assert torch.equal(padded[:, 2976:, :],
                     padded[:, -1:, :].expand(1, 25, CLASSES_NUM))


def test_interpolate_repeats_each_frame():
  '''The upsample is interleaved: every frame is repeated `ratio` times.'''
  frames = torch.tensor([[[1.0], [2.0], [3.0]]])
  upsampled = interpolate(frames, 2)
  assert upsampled.shape == (1, 6, 1)
  assert upsampled.flatten().tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_apply_aug_false_with_a_mixup_lambda_does_not_halve():
  '''The original halves the batch even when the mixup did not run.'''
  pytest.importorskip('birdcall.augmentations.mixup')
  model = _build_model(apply_aug=False)
  model.train()
  waveform = torch.randn(4, 1, N_SAMPLES)
  mixup_lambda = torch.full((4,), 0.5)
  out = model((waveform, mixup_lambda))
  assert out['clipwise_output'].shape == (4, 1, CLASSES_NUM)
  assert out['framewise_output'].shape == (4, 1, FRAMES, CLASSES_NUM)


def test_mixup_halves_the_batch():
  '''With the augmentation on, the batch axis changes meaning.'''
  pytest.importorskip('birdcall.augmentations.mixup')
  model = _build_model(apply_aug=True)
  model.train()
  waveform = torch.randn(4, 1, N_SAMPLES)
  mixup_lambda = torch.full((4,), 0.5)
  out = model((waveform, mixup_lambda))
  assert out['clipwise_output'].shape == (2, 1, CLASSES_NUM)
  assert out['framewise_output'].shape == (2, 1, FRAMES, CLASSES_NUM)


def test_front_end_buffers_are_not_parameters(model):
  '''The frozen front-end is a buffer, not a trainable parameter.'''
  parameter_names = {name for name, _ in model.named_parameters()}
  buffer_names = {name for name, _ in model.named_buffers()}
  assert 'spectrogram_extractor.stft.fft_window' in buffer_names
  assert 'spectrogram_extractor.stft.fft_window' not in parameter_names
  # The mel matrix is a frozen PARAMETER in the original, not a buffer.
  assert 'logmel_extractor.melW' in parameter_names
  assert 'logmel_extractor.melW' not in buffer_names
  assert not dict(model.named_parameters())['logmel_extractor.melW'] \
      .requires_grad
  for name, buffer in model.named_buffers():
    if name == 'spectrogram_extractor.stft.fft_window':
      assert not buffer.requires_grad
  # The STFT convolutions are frozen parameters, as in the original.
  for name, parameter in model.named_parameters():
    if name.startswith('spectrogram_extractor'):
      assert not parameter.requires_grad


def test_the_preprocess_returns_the_log_mel_and_the_frame_count(model):
  '''The front-end runs inside the graph, and reports its frame count.'''
  waveform = torch.randn(2, N_SAMPLES)
  x, frames_num = model.preprocess(waveform)
  assert frames_num == FRAMES
  assert x.shape == (2, 1, FRAMES, MEL_BINS)


def test_the_mixup_is_only_active_in_training_mode_with_the_augmentation():
  '''The mixup needs the training mode, the switch and a lambda.'''
  model = _build_model(apply_aug=True)
  assert model._mixup_active(torch.full((2,), 0.5)) is False
  model.train()
  assert model._mixup_active(None) is False
  model.apply_aug = False
  assert model._mixup_active(torch.full((2,), 0.5)) is False
  model.apply_aug = True
  assert model._mixup_active(torch.full((2,), 0.5)) is True
