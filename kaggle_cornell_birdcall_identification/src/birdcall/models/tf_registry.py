#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The model registry of the Keras backend.

Only the DenseNet-121 variant is shipped: it is the model all 13
ensemble members use.  The other four PANNs variants have no shipped
config that selects them, and ``keras.applications`` has no DenseNet
161, 169 or 201 anyway.
'''


import tensorflow as tf
from birdcall.models.tf_model_panns_dense121_att import PANNsDense121Att


MODEL_REGISTRY: dict[str, type] = {
    'sed_dense121att': PANNsDense121Att,
}

def build_model(name: str, **kwargs) -> tf.keras.models.Model:
  '''Builds a model from the registry.

  Args:
    name: The registry key, ``'sed_dense121att'``.
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
