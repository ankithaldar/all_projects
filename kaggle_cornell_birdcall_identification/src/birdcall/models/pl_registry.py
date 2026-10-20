#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The model registry.

Only the DenseNet-121 variant is shipped: it is the model all 13
ensemble members use.  The other four PANNs variants have no shipped
config that selects them.
'''

from torch import nn

from birdcall.models.pl_model_panns_dense121_att import PANNsDense121Att

MODEL_REGISTRY: dict[str, type] = {
    'sed_dense121att': PANNsDense121Att,
}


def build_model(name: str, **kwargs) -> nn.Module:
  '''Builds a model from the registry.

  Args:
    name: The registry key, `'sed_dense121att'`.
    **kwargs: The keyword arguments of the model constructor.

  Returns:
    The model.

  Raises:
    ValueError: If the name is not in the registry.
  '''
  if name not in MODEL_REGISTRY:
    raise ValueError(
        f'Unknown model {name!r}. Valid names: '
        f'{sorted(MODEL_REGISTRY)}.')
  return MODEL_REGISTRY[name](**kwargs)
