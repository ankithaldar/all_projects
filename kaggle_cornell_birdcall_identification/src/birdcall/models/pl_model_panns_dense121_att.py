#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The PANNs DenseNet-121 attention model.

The whole front-end runs inside the graph: waveform in, log-mel out,
every step.  The backbone is the ImageNet-pretrained
`torchvision.models.densenet121`, and the head is the `AttBlock`, which
reads the same classifier at clip level and at frame level.

`init_layer` and `init_bn` are re-exported from
`birdcall.models.pl_model_att_block`, where they are defined, because
the `AttBlock` uses them too.
'''

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from birdcall.augmentations.mixup import do_mixup
from birdcall.helpers.pl_frontend_logmel import LogmelFilterBank
from birdcall.helpers.pl_frontend_spec_augment import SpecAugmentation
from birdcall.helpers.pl_frontend_spectrogram import Spectrogram
from birdcall.models.pl_model_att_block import AttBlock
from birdcall.models.pl_model_att_block import init_bn, init_layer


def interpolate(x: torch.Tensor, ratio: int) -> torch.Tensor:
  '''Upsamples the time axis of a framewise output.

  The upsample is an interleaved nearest-neighbour one: every frame is
  repeated `ratio` times consecutively.  It compensates the x32 time
  downsampling of the backbone.

  Args:
    x: A `(B, n_frames, n_classes)` framewise output.
    ratio: The number of copies of each frame.

  Returns:
    A `(B, n_frames * ratio, n_classes)` tensor.
  '''
  batch_size, time_steps, classes_num = x.shape
  x = x[:, :, None, :].repeat(1, 1, ratio, 1)
  return x.reshape(batch_size, time_steps * ratio, classes_num)


def pad_framewise_output(framewise_output: torch.Tensor,
                         frames_num: int) -> torch.Tensor:
  '''Pads a framewise output to the requested number of frames.

  The padding value is the value of the last frame, so the clip-level
  constant carries through to the end of the file instead of a hard
  "no bird here" discontinuity.

  Args:
    framewise_output: A `(B, n_frames, n_classes)` framewise output.
    frames_num: The number of frames to pad to.

  Returns:
    A `(B, frames_num, n_classes)` tensor.
  '''
  pad = framewise_output[:, -1:, :].repeat(
      1, frames_num - framewise_output.shape[1], 1)
  return torch.cat((framewise_output, pad), dim=1)


class PANNsDense121Att(nn.Module):
  '''The DenseNet-121 sound event detection model with attention.'''

  def __init__(self, sample_rate: int, window_size: int, hop_size: int,
               mel_bins: int, fmin: int, fmax: int, classes_num: int,
               apply_aug: bool, top_db: float | None = None) -> None:
    '''Builds the model.

    Args:
      sample_rate: The audio sample rate in Hz.
      window_size: The FFT size, and the analysis window length.
      hop_size: The hop between frames, in samples.
      mel_bins: The number of mel bands.
      fmin: The lowest frequency in Hz.
      fmax: The highest frequency in Hz.
      classes_num: The number of classes.
      apply_aug: The kill switch for both SpecAugment and mixup.
      top_db: The dynamic range ceiling of the log-mel, in dB.  `None`
        disables it, which is what every shipped config does.
    '''
    super().__init__()
    self.sample_rate = sample_rate
    self.window_size = window_size
    self.hop_size = hop_size
    self.mel_bins = mel_bins
    self.classes_num = classes_num
    self.apply_aug = apply_aug
    # The x32 time downsampling of the DenseNet backbone.
    self.interpolate_ratio = 32

    self.spectrogram_extractor = Spectrogram(
        n_fft=window_size, hop_length=hop_size, win_length=window_size,
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
    self.logmel_extractor = LogmelFilterBank(
        sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
        fmax=fmax, ref=1.0, amin=1e-10, top_db=top_db,
        freeze_parameters=True)
    self.spec_augmenter = SpecAugmentation(time_drop_width=64,
                                           time_stripes_num=2,
                                           freq_drop_width=8,
                                           freq_stripes_num=2)
    self.bn0 = nn.BatchNorm2d(mel_bins)
    self.fc1 = nn.Linear(1024, 1024, bias=True)
    self.att_block = AttBlock(1024, classes_num, activation='sigmoid')
    self.densenet_features = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
    self.init_weight()

  def init_weight(self) -> None:
    '''Initialises the from-scratch layers.

    The DenseNet backbone is deliberately left alone: re-initialising it
    would destroy the ImageNet pretraining, which is the whole point of
    the DenseNet swap.
    '''
    init_layer(self.fc1)
    init_bn(self.bn0)
    self.att_block.init_weights()

  def _mixup_active(self, mixup_lambda: torch.Tensor | None) -> bool:
    '''Whether the mixup runs on this forward pass.

    Args:
      mixup_lambda: The mixup coefficients, or `None`.

    Returns:
      True when the spectrogram is about to be mixed.
    '''
    return bool(self.training and self.apply_aug and
                mixup_lambda is not None)

  def preprocess(self, input_x: torch.Tensor,
                 mixup_lambda: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, int]:
    '''Turns a waveform batch into a normalised log-mel batch.

    The `bn0` sandwich -- transpose, normalise over the mel bins,
    transpose back -- puts the mel bins on the channel axis, which is
    what `BatchNorm2d` normalises over, and it is load bearing: the
    stripe dropper writes a literal zero, which after the batch norm is
    the per-bin mean, "nothing was here" in the model's own units.

    Args:
      input_x: A `(B * C, n_samples)` waveform batch.
      mixup_lambda: The mixup coefficients, or `None`.

    Returns:
      A tuple `(x, frames_num)`, the log-mel and the number of frames.
    '''
    x = self.spectrogram_extractor(input_x)
    x = self.logmel_extractor(x)
    frames_num = x.shape[2]

    x = x.transpose(1, 3)
    x = self.bn0(x)
    x = x.transpose(1, 3)

    if self.training and self.apply_aug:
      x = self.spec_augmenter(x)

    if self._mixup_active(mixup_lambda):
      x = do_mixup(x, mixup_lambda)

    return x, frames_num

  def forward(self, inputs: tuple[torch.Tensor, torch.Tensor | None]
              ) -> dict[str, torch.Tensor]:
    '''Runs the model.

    The batch axis changes meaning when the mixup runs: the mixed
    spectrograms are `(B * C) / 2` rows, which are reinterpreted as
    `B / 2` files of one snippet each.  The original halves the batch
    whenever `mixup_lambda is not None`, even when `apply_aug` is off,
    which crashes on the reshape; here the halving is guarded by the
    same condition that runs the mixup.

    Args:
      inputs: A tuple `(input_x, mixup_lambda)`.  `input_x` is a
        `(B, C, n_samples)` waveform batch, and `mixup_lambda` is a
        `(B * C,)` tensor of coefficients, or `None`.

    Returns:
      A dict with `clipwise_output`, of shape `(B, C, n_classes)`, and
      `framewise_output`, of shape `(B, C, n_frames, n_classes)`.
    '''
    input_x, mixup_lambda = inputs
    b, c, s = input_x.shape
    input_x = input_x.reshape(b * c, s)

    mixed = self._mixup_active(mixup_lambda)
    x, frames_num = self.preprocess(input_x, mixup_lambda=mixup_lambda)
    if mixed:
      b = (b * c) // 2
      c = 1

    # A view, zero copy: the same log-mel on three channels.
    x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])
    x = self.densenet_features(x)
    # The frequency axis has collapsed to 2 bins by now.
    x = torch.mean(x, dim=3)
    x = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
    x = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
    x = F.dropout(x, p=0.5, training=self.training)
    # The original transposes around the projection, because its fc1 is an
    # nn.Linear and therefore mixes the last axis.
    x = x.transpose(1, 2)
    x = F.relu(self.fc1(x))
    x = x.transpose(1, 2)
    x = F.dropout(x, p=0.5, training=self.training)

    clipwise_output, _, segmentwise_output = self.att_block(x)
    framewise_output = segmentwise_output.transpose(1, 2)
    framewise_output = interpolate(framewise_output, self.interpolate_ratio)
    framewise_output = pad_framewise_output(framewise_output, frames_num)

    return {
        'clipwise_output': clipwise_output.reshape(b, c, self.classes_num),
        'framewise_output': framewise_output.reshape(b, c, frames_num,
                                                     self.classes_num),
    }
