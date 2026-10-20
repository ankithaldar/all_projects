#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the Keras model registry.

Only the DenseNet-121 variant is shipped: it is the model all thirteen
ensemble members use.  An unknown model name is a clear error, not an
`AttributeError`.
'''

import pytest

from birdcall.models.tf_registry import MODEL_REGISTRY
from birdcall.models.tf_registry import build_model


def test_the_model_registry_holds_the_one_shipped_model() -> None:
  '''Only the DenseNet-121 variant is shipped.'''
  assert list(MODEL_REGISTRY) == ['sed_dense121att']
  assert MODEL_REGISTRY['sed_dense121att'].__name__ == 'PANNsDense121Att'


def test_an_unknown_model_raises() -> None:
  '''An unknown model name is a clear error, not an AttributeError.'''
  with pytest.raises(ValueError, match='Unknown model'):
    build_model('sed_ccnn14att')
