#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The log-mel filterbank.

`LogmelFilterBank` projects the linear power spectrogram onto the Slaney
mel scale and converts the result to decibels.

The mel matrix is the committed Slaney (area normalised) matrix from
`birdcall.helpers.mel_matrix`, not a runtime `librosa.filters.mel` call:
librosa is not a runtime dependency, and `tf.signal`'s matrix is HTK,
which is a *different* matrix.

The dB conversion is forced to fp32.  The `amin = 1e-10` floor underflows
to zero in fp16, `log10(0)` is `-inf`, and the resulting NaNs are the
reason the original hard-disabled AMP.
'''

import math

import torch
from torch import nn

from birdcall.helpers.mel_matrix import load_mel_matrix


class LogmelFilterBank(nn.Module):
  '''The mel projection and the power-to-decibel conversion.'''

  def __init__(self, sr: int = 32000, n_fft: int = 2048, n_mels: int = 64,
               fmin: int = 50, fmax: int = 14000, is_log: bool = True,
               ref: float = 1.0, amin: float = 1e-10,
               top_db: float | None = 80.0,
               freeze_parameters: bool = True) -> None:
    '''Builds the frozen mel filterbank.

    Args:
      sr: The audio sample rate in Hz.
      n_fft: The FFT size of the incoming spectrogram.
      n_mels: The number of mel bands.
      fmin: The lowest frequency in Hz.
      fmax: The highest frequency in Hz.
      is_log: Whether to convert the mel spectrum to decibels.
      ref: The reference power, in dB.  `1.0` gives absolute dB.
      amin: The floor of the power, before the logarithm.
      top_db: The dynamic range ceiling, in dB, relative to the maximum
        of the batch.  `None` disables the clamp, which is what every
        shipped config does.
      freeze_parameters: Whether to freeze the mel matrix.

    Raises:
      ValueError: If `top_db` is negative.
    '''
    super().__init__()
    if top_db is not None and top_db < 0:
      raise ValueError('top_db must be non-negative')
    self.sr = sr
    self.n_fft = n_fft
    self.n_mels = n_mels
    self.fmin = fmin
    self.fmax = fmax
    self.is_log = is_log
    self.ref = ref
    self.amin = amin
    self.top_db = top_db
    self.freeze_parameters = freeze_parameters

    mel = load_mel_matrix(sr, n_fft, n_mels, fmin, fmax)
    # (n_fft // 2 + 1, n_mels), so the mel axis is the last one and the
    # projection is a plain matmul on it.  The name matches the original
    # checkpoint key, `logmel_extractor.melW`.
    #
    # The original holds the matrix as an nn.Parameter with
    # requires_grad=False, not as a buffer, and that matters: it is part
    # of the checkpoint's state_dict under that name, and it is counted
    # by `model.parameters()` even though it never receives a gradient.
    self.melW = nn.Parameter(torch.from_numpy(mel),
                             requires_grad=not freeze_parameters)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    '''Projects a spectrogram onto the mel scale.

    Args:
      input: A `(B, C, n_frames, n_fft // 2 + 1)` power spectrogram, or
        a `(B, n_fft // 2 + 1, n_frames)` one.

    Returns:
      The log-mel spectrogram, with the same leading axes and
      `n_mels` as the last one: `(B, C, n_frames, n_mels)` for the 4-D
      input, `(B, n_frames, n_mels)` for the 3-D one.
    '''
    # The whole projection is done in fp32: the dB conversion below is
    # not safe in fp16, and the matmul is cheap enough not to matter.
    with torch.amp.autocast(device_type=input.device.type, enabled=False):
      x = input.float()
      if x.ndim == 3:
        # (B, n_bins, n_frames) -> (B, n_frames, n_bins)
        x = x.transpose(1, 2)
      mel_spectrogram = torch.matmul(x, self.melW)
      if self.is_log:
        return self.power_to_db(mel_spectrogram)
      return mel_spectrogram

  def power_to_db(self, input: torch.Tensor) -> torch.Tensor:
    '''Converts a mel power spectrogram to decibels.

    Args:
      input: The mel power spectrogram.

    Returns:
      The same tensor in decibels, in fp32.
    '''
    with torch.amp.autocast(device_type=input.device.type, enabled=False):
      x = input.float()
      log_spec = 10.0 * torch.log10(torch.clamp(x, min=self.amin))
      log_spec = log_spec - 10.0 * math.log10(max(self.amin, self.ref))
      if self.top_db is not None:
        log_spec = torch.clamp(log_spec,
                               min=log_spec.max().item() - self.top_db)
    return log_spec
