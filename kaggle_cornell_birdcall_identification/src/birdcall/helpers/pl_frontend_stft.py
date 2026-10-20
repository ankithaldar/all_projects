#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The STFT, implemented as two frozen `Conv1d` layers.

The original vendored `torchlibrosa.stft.STFT`: the analysis window is
folded into the DFT matrix, so one convolution performs the windowing
*and* the transform.  The real and the imaginary parts are two separate
convolutions with `bias=False` and frozen weights.

Two deviations from the original, both deliberate:

- The window is built with `torch.hann_window` instead of
  `librosa.filters.get_window`, so the module has no librosa dependency
  at runtime.  `torch.hann_window(periodic=True)` is the same periodic
  Hann that librosa returns for `fftbins=True`.
- The original centred the window with
  `librosa.util.pad_center(fft_window, n_fft)`, passing `size`
  positionally.  librosa 0.10 made `size` keyword-only, so that call
  raises `TypeError` on any modern librosa and the original model
  cannot be constructed at all.  The centring is a plain zero pad here.
'''

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _dft_matrix(n_fft: int) -> np.ndarray:
  '''Builds the analysis DFT matrix.

  Args:
    n_fft: The FFT size.

  Returns:
    A ``(n_fft, n_fft)`` complex64 matrix whose element ``(i, j)`` is
    ``exp(-2j * pi * i * j / n_fft)``.
  '''
  index = np.arange(n_fft)
  return np.exp(-2j * np.pi * np.outer(index, index) / n_fft).astype(
      np.complex64)


def _pad_center(window: np.ndarray, size: int) -> np.ndarray:
  '''Centres a window in a zero-padded array of the given size.

  This is what `librosa.util.pad_center(data, size=size)` does with its
  default `mode='constant'`.  The original called it positionally, which
  is fatal on librosa >= 0.10.

  Args:
    window: The 1-D window to pad.
    size: The padded length.

  Returns:
    The centred window, of length `size`.
  '''
  pad = size - window.shape[-1]
  if pad <= 0:
    return window
  left = pad // 2
  return np.pad(window, (left, pad - left))


class STFT(nn.Module):
  '''The short-time Fourier transform, as two frozen convolutions.

  The weights are the DFT matrix times the analysis window, so the
  transform runs inside the autograd graph.  `center=True` reflect-pads
  the waveform by `n_fft // 2` samples on both sides first, which is
  what makes the frame count `1 + n_samples // hop_length`.
  '''

  def __init__(self, n_fft: int = 2048, hop_length: int | None = None,
               win_length: int | None = None, window: str | None = 'hann',
               center: bool = True, pad_mode: str = 'reflect',
               freeze_parameters: bool = True) -> None:
    '''Builds the frozen DFT-window filters.

    Args:
      n_fft: The FFT size, and the number of linear frequency bins
        produced is `n_fft // 2 + 1`.
      hop_length: The hop between frames, in samples.  Defaults to
        `win_length // 4`.
      win_length: The length of the analysis window, in samples.
        Defaults to `n_fft`.
      window: The window name.  Only `'hann'` (and `None`, for a
        rectangular window) is supported.
      center: Whether to reflect-pad the input by `n_fft // 2`.
      pad_mode: The padding mode, `'reflect'` or `'constant'`.
      freeze_parameters: Whether to freeze the convolution weights.

    Raises:
      ValueError: If the window or the pad mode is not supported.
    '''
    super().__init__()
    if win_length is None:
      win_length = n_fft
    if hop_length is None:
      hop_length = win_length // 4
    if pad_mode not in ('constant', 'reflect'):
      raise ValueError(f'Unsupported pad_mode: {pad_mode!r}')
    if window not in ('hann', None):
      raise ValueError(f'Unsupported window: {window!r}')

    self.n_fft = n_fft
    self.win_length = win_length
    self.hop_length = hop_length
    self.window = window
    self.center = center
    self.pad_mode = pad_mode
    self.freeze_parameters = freeze_parameters

    dft = _dft_matrix(n_fft)
    # Not persistent: the DFT matrix is not part of the original
    # checkpoint, and it is 8 MB of dead weight after the build.
    self.register_buffer('dft_matrix', torch.from_numpy(dft),
                         persistent=False)

    if window is None:
      fft_window = np.ones(n_fft, dtype=np.float32)
    else:
      fft_window = torch.hann_window(
          win_length, periodic=True).numpy().astype(np.float32)
    fft_window = _pad_center(fft_window, n_fft)
    self.register_buffer('fft_window', torch.from_numpy(fft_window),
                         persistent=False)

    # (n_fft, n_fft // 2 + 1): element (n, k) is W[n, k] * window[n].
    kernel = dft[:, :n_fft // 2 + 1] * fft_window[:, None]
    self.conv_real = nn.Conv1d(1, n_fft // 2 + 1, kernel_size=n_fft,
                               stride=hop_length, bias=False)
    self.conv_imag = nn.Conv1d(1, n_fft // 2 + 1, kernel_size=n_fft,
                               stride=hop_length, bias=False)
    with torch.no_grad():
      self.conv_real.weight.copy_(
          torch.from_numpy(
              np.ascontiguousarray(kernel.real.T)).unsqueeze(1))
      self.conv_imag.weight.copy_(
          torch.from_numpy(
              np.ascontiguousarray(kernel.imag.T)).unsqueeze(1))
    if freeze_parameters:
      for parameter in self.parameters():
        parameter.requires_grad = False

  def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''Transforms a batch of waveforms.

    Args:
      input: A `(B, n_samples)` waveform batch.

    Returns:
      A tuple `(real, imag)`, each `(B, n_fft // 2 + 1, n_frames)`.
    '''
    x = input[:, None, :]
    if self.center:
      x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2),
                mode=self.pad_mode)
    return self.conv_real(x), self.conv_imag(x)
