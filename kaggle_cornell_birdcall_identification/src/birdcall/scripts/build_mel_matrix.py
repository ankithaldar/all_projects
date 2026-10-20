#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Rebuild the committed Slaney mel matrix.

The mel filterbank matrix depends on four integers and on librosa's
defaults, and librosa is an awkward runtime dependency: it drags in
numba, which pins numpy.  The matrix is therefore built **once**, at
build time, and committed as a ``.npy`` next to the package.  Neither
backend needs librosa at runtime.

This script is that build step.  It needs librosa, and it fails with a
clear message when librosa is missing, because "librosa is not
installed" is the expected answer in a runtime environment and a
confusing one here.

Usage:

    python -m birdcall.scripts.build_mel_matrix
'''

import argparse
from pathlib import Path
from typing import Optional

from birdcall.helpers.mel_matrix import build_mel_matrix
from birdcall.helpers.mel_matrix import save_mel_matrix

# The shipped values.  They are the ones the committed matrix was built
# with, and the ones every configuration selects.
DEFAULT_SAMPLE_RATE = 32000
DEFAULT_N_FFT = 1024
DEFAULT_N_MELS = 64
DEFAULT_FMIN = 50
DEFAULT_FMAX = 14000

MISSING_LIBROSA = (
    'librosa is not installed. It is a build-time only dependency: '
    'install it with `pip install "birdcall[build]"`, or use a runtime '
    'environment that reads the committed matrix.')


def build(sample_rate: int, n_fft: int, n_mels: int, fmin: int,
          fmax: int) -> Path:
  '''Builds the mel matrix and writes it to the assets directory.

  Args:
    sample_rate: The audio sample rate in Hz.
    n_fft: The FFT size.
    n_mels: The number of mel bands.
    fmin: The lowest frequency in Hz.
    fmax: The highest frequency in Hz.

  Returns:
    The path the matrix was written to.

  Raises:
    SystemExit: If librosa is not installed.
  '''
  try:
    matrix = build_mel_matrix(sample_rate, n_fft, n_mels, fmin, fmax)
  except ImportError as error:
    raise SystemExit(MISSING_LIBROSA) from error
  return save_mel_matrix(matrix, sample_rate, n_fft, n_mels, fmin, fmax)


def build_parser() -> argparse.ArgumentParser:
  '''Builds the command line parser.

  Returns:
    The parser.
  '''
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--sample-rate', type=int, default=DEFAULT_SAMPLE_RATE,
                      help='The audio sample rate in Hz.')
  parser.add_argument('--n-fft', type=int, default=DEFAULT_N_FFT,
                      help='The FFT size.')
  parser.add_argument('--n-mels', type=int, default=DEFAULT_N_MELS,
                      help='The number of mel bands.')
  parser.add_argument('--fmin', type=int, default=DEFAULT_FMIN,
                      help='The lowest frequency of the mel bank in Hz.')
  parser.add_argument('--fmax', type=int, default=DEFAULT_FMAX,
                      help='The highest frequency of the mel bank in Hz.')
  return parser


def main(argv: Optional[list[str]] = None) -> int:
  '''Rebuilds the mel matrix.

  Args:
    argv: The command line arguments, or ``None`` for the real ones.

  Returns:
    The exit code.
  '''
  args = build_parser().parse_args(argv)
  path = build(args.sample_rate, args.n_fft, args.n_mels, args.fmin,
               args.fmax)
  print(f'Wrote {path}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
