#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The mixup coefficient generator and the mixing kernel.

The original drew one ``Beta(alpha, alpha)`` per **pair** and emitted
``[lam, 1 - lam]``, so the even and the odd rows of the batch are
partners.  The hard-coded ``random_seed=1234`` is kept as the default:
the docs ask for a faithful port, because the seed changes the
experiment.

``do_mixup`` is not swap-based.  It is a lambda-weighted lerp between
adjacent rows, and it is rank-agnostic, so the very same function mixes
a ``(N, 264)`` label matrix and a ``(N, 1, 3001, 64)`` log-mel batch.
The original squeezed the lambda into the trailing axes with a
``transpose(0, -1)`` sandwich; that idiom is torch-specific -- numpy
reads ``transpose(0, -1)`` as "keep the axes where they are" -- so the
kernel reshapes the lambda instead.  The result is the same, on both.

``torch`` is resolved by name, when the coefficients are drawn, so that
importing this module does not pull a framework in; the mixing kernel
itself is duck-typed and works on a numpy array as well.
'''

import importlib

import numpy as np


class Mixup:
  '''Draws the mixup coefficients of a batch.

  Attributes:
    mixup_alpha: The shape parameter of the Beta distribution.
    random_state: The private generator the coefficients are drawn from.
  '''

  def __init__(self, mixup_alpha: float, random_seed: int = 1234) -> None:
    '''Builds the coefficient generator.

    Args:
      mixup_alpha: The shape parameter of the Beta distribution.  The
        shipped configs use ``1.0``, which is a flat uniform draw.
      random_seed: The seed of the private generator.  The original
        hard-coded ``1234`` and it is kept as the default.
    '''
    self.mixup_alpha = mixup_alpha
    self.random_state = np.random.RandomState(random_seed)

  def get_lambda(self, batch_size: int, device) -> 'torch.Tensor':
    '''Draws the coefficients of one batch.

    One ``Beta(alpha, alpha)`` is drawn per pair and emitted as
    ``[lam, 1 - lam]``, so adjacent coefficients sum to one.

    Args:
      batch_size: The number of rows of the batch.  It is rounded up to
        an even number of coefficients.
      device: The device the coefficients are placed on.

    Returns:
      A float32 tensor of ``batch_size`` coefficients.
    '''
    torch = importlib.import_module('torch')

    mixup_lambdas = []
    for _ in range(0, batch_size, 2):
      lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
      mixup_lambdas.append(lam)
      mixup_lambdas.append(1. - lam)
    return torch.tensor(mixup_lambdas,
                        requires_grad=False).float().to(device)


def do_mixup(x, mixup_lambda):
  '''Mixes the rows of a batch with their neighbour.

  Args:
    x: The batch, ``(N, ...)`` with an even ``N``.
    mixup_lambda: The coefficients, ``(N,)``.

  Returns:
    ``x[0::2] * mixup_lambda[0::2] + x[1::2] * mixup_lambda[1::2]``, with
    the batch halved to ``(N // 2, ...)``.  The kernel is rank-agnostic:
    it mixes a label matrix and a log-mel batch alike.
  '''
  even = x[0::2]
  odd = x[1::2]
  shape = (-1,) + (1,) * (even.ndim - 1)
  lam_even = mixup_lambda[0::2].reshape(shape)
  lam_odd = mixup_lambda[1::2].reshape(shape)
  return even * lam_even + odd * lam_odd
