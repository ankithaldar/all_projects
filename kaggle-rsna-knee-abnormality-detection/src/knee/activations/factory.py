#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Activation factory shared by layers and models.

Single source of truth so YAML can select activations by name without
torch-specific branching at call sites (Simple Factory pattern).
"""

from __future__ import annotations

from torch import nn

_REGISTRY: dict[str, type[nn.Module]] = {
  'relu': nn.ReLU,
  'gelu': nn.GELU,
  'silu': nn.SiLU,
  'mish': nn.Mish,
}


def get_activation(name: str = 'silu') -> nn.Module:
  """Build an activation module by name.

  Args:
      name: One of 'relu', 'gelu', 'silu', 'mish' (case-insensitive).

  Returns:
      Instantiated activation module.

  Raises:
      ValueError: If the name is not registered.
  """
  key = name.lower()
  if key not in _REGISTRY:
    raise ValueError(
      f"unknown activation '{name}'; options: {sorted(_REGISTRY)}"
    )
  return _REGISTRY[key]()
