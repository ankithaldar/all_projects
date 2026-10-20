#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The registry of the five reachable SED criteria, for the Keras backend.

The engine selects a criterion by the string ``criterion_name`` and
instantiates it with ``criterion_params``.  This module is the dispatch
table for that selection: the names are the ones the configs use, and
they are the same constants the configuration layer validates against.

Only these five criteria are reachable from the SED path.  The other
eleven files of the original's loss zoo are dead research history and
are not ported.
'''


import tensorflow as tf
from birdcall.config_params.criterion_params import (
    CRITERION_BCE_CONTROLLED_STANDARD)
from birdcall.config_params.criterion_params import CRITERION_FOCAL_STANDARD
from birdcall.config_params.criterion_params import CRITERION_REMOVED_FOCAL
from birdcall.config_params.criterion_params import (
    CRITERION_SCALED_POS_NEG_FOCAL)
from birdcall.config_params.criterion_params import (
    CRITERION_SCALED_POS_NEG_FOCAL_AUGD)
from birdcall.losses.tf_loss_bce_controlled_standard import (
    BCEControlledStandardLoss)
from birdcall.losses.tf_loss_focal_standard import FocalLossStandard
from birdcall.losses.tf_loss_removed_focal import SedRemovedFocalLoss
from birdcall.losses.tf_loss_scaled_pos_neg_focal import (
    SedScaledPosNegFocalLoss)
from birdcall.losses.tf_loss_scaled_pos_neg_focal_augd import (
    SedScaledPosNegFocalLossAugd)


CRITERION_REGISTRY: dict[str, type] = {
    CRITERION_FOCAL_STANDARD: FocalLossStandard,
    CRITERION_REMOVED_FOCAL: SedRemovedFocalLoss,
    CRITERION_SCALED_POS_NEG_FOCAL: SedScaledPosNegFocalLoss,
    CRITERION_SCALED_POS_NEG_FOCAL_AUGD: SedScaledPosNegFocalLossAugd,
    CRITERION_BCE_CONTROLLED_STANDARD: BCEControlledStandardLoss,
}

def build_criterion(name: str, **params) -> tf.keras.losses.Loss:
  '''Builds the criterion of the given name.

  Args:
    name: One of the keys of ``CRITERION_REGISTRY``.
    **params: The hyper-parameters of the criterion, for example
      ``gamma``, ``alpha_0``, ``alpha_1`` and ``secondary_factor``.

  Returns:
    The criterion.

  Raises:
    ValueError: If the name is not one of the registered criteria.
  '''
  if name not in CRITERION_REGISTRY:
    raise ValueError(
        f'Unknown criterion {name!r}. '
        f'Valid names: {sorted(CRITERION_REGISTRY)}')
  return CRITERION_REGISTRY[name](**params)
