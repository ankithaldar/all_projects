#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The attention head.

`AttBlock` produces the two readings the competition needs from one
tensor: a clip-level score, pooled over time with a learned attention,
and a frame-level score, which is the same classifier read per frame.

The softmax is over **time**.  `axis=-1` on a `(B, classes, n_frames)`
tensor is the time axis in PyTorch, but it is the class axis in Keras,
where it is a silent wrong answer.
'''

import torch
from torch import nn


def init_layer(layer: nn.Module) -> None:
  '''Initialises a layer with Xavier/Glorot uniform weights and no bias.

  Args:
    layer: The layer to initialise.
  '''
  nn.init.xavier_uniform_(layer.weight)
  if hasattr(layer, 'bias') and layer.bias is not None:
    nn.init.constant_(layer.bias, 0.0)


def init_bn(bn: nn.Module) -> None:
  '''Resets a batch norm to the identity.

  Args:
    bn: The batch norm to initialise.
  '''
  nn.init.constant_(bn.bias, 0.0)
  nn.init.constant_(bn.weight, 1.0)


class AttBlock(nn.Module):
  '''The clip-level and frame-level heads, sharing one classifier.'''

  def __init__(self, in_features: int, out_features: int,
               activation: str = 'linear', temperature: float = 1.0) -> None:
    '''Builds the attention block.

    Args:
      in_features: The number of input channels, 1024 for the
        DenseNet-121 variant.
      out_features: The number of classes.
      activation: The activation applied to the classifier output,
        `'sigmoid'` or `'linear'`.
      temperature: Dead in the original, and dead here.  The upstream
        `softmax(x / temperature)` was never written, so the value is
        accepted and ignored.
    '''
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.activation = activation
    self.att = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
    self.cla = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
    # Dead in the original too: `bn_att` is created and initialised but
    # never used in `forward`.  It is kept because the original
    # checkpoint carries it, and the parameter count is part of the
    # documented contract.
    self.bn_att = nn.BatchNorm1d(out_features)
    self.temperature = temperature

  def init_weights(self) -> None:
    '''Initialises the classifier and the dead batch norm.'''
    init_layer(self.att)
    init_layer(self.cla)
    init_bn(self.bn_att)

  def nonlinear_transform(self, x: torch.Tensor) -> torch.Tensor:
    '''Applies the activation.

    Args:
      x: The classifier output.

    Returns:
      The activated output.

    Raises:
      ValueError: If the activation is not supported.
    '''
    if self.activation == 'linear':
      return x
    if self.activation == 'sigmoid':
      return torch.sigmoid(x)
    raise ValueError(f'Unknown activation: {self.activation!r}')

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor]:
    '''Reads a feature tensor at both granularities.

    Args:
      x: A `(B, in_features, n_frames)` feature tensor.

    Returns:
      A tuple `(clipwise, norm_att, cla)`.  `clipwise` is
      `(B, out_features)`, the attention-weighted sum over time.
      `norm_att` is `(B, out_features, n_frames)` and sums to one over
      the time axis.  `cla` is `(B, out_features, n_frames)`, the
      activated classifier output.
    '''
    # The softmax is over the time axis, which is the last one here.
    norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
    cla = self.nonlinear_transform(self.cla(x))
    out = torch.sum(norm_att * cla, dim=2)
    return out, norm_att, cla
