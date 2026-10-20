#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The per-criterion hyper-parameter schemas.

Each of the five reachable criteria takes a different set of parameters.
The original passed a plain dict straight into the criterion constructor, so
a parameter meant for one criterion was silently accepted by another.  Here
each criterion has its own schema, and the selected schema is used to
validate the parameters of the selected criterion.

The values are the original's shipped ones.  Note that with
``gamma=0``, ``alpha_0=alpha_1=1`` and ``secondary_factor=1`` the scaled
focal loss is **bit-identical to mean BCE**: the focal term is inert, and
that is the objective that won.  It is reproduced deliberately.
'''

from typing import Type

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

CRITERION_FOCAL_STANDARD = 'focal_loss_standard'
CRITERION_REMOVED_FOCAL = 'sed_removed_focal_loss'
CRITERION_SCALED_POS_NEG_FOCAL = 'sed_scaled_pos_neg_focal_loss'
CRITERION_SCALED_POS_NEG_FOCAL_AUGD = 'sed_scaled_pos_neg_focal_loss_augd'
CRITERION_BCE_CONTROLLED_STANDARD = 'bce_controlled_standard_loss'

CRITERION_NAMES = (
    CRITERION_FOCAL_STANDARD,
    CRITERION_REMOVED_FOCAL,
    CRITERION_SCALED_POS_NEG_FOCAL,
    CRITERION_SCALED_POS_NEG_FOCAL_AUGD,
    CRITERION_BCE_CONTROLLED_STANDARD,
)


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class FocalLossStandardParams:
  '''The parameters of ``FocalLossStandard``.

  Attributes:
    gamma: The focal exponent.  Zero makes the focal term inert.
    alpha: The global scale of the focal term.
  '''

  gamma: float = 0.0
  alpha: float = 1.0


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class SedRemovedFocalLossParams:
  '''The parameters of ``SedRemovedFocalLoss``.

  Attributes:
    gamma: The focal exponent.
    alpha: The global scale of the focal term.
  '''

  gamma: float = 0.0
  alpha: float = 1.0


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class SedScaledPosNegFocalLossParams:
  '''The parameters of ``SedScaledPosNegFocalLoss``.

  Attributes:
    gamma: The focal exponent.
    alpha_0: The scale of the negative positions.
    alpha_1: The scale of the positive positions.
    secondary_factor: The scale of the secondary species.  One disables
      the secondary masking.
  '''

  gamma: float = 0.0
  alpha_0: float = 1.0
  alpha_1: float = 1.0
  secondary_factor: float = 0.1


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class SedScaledPosNegFocalLossAugdParams:
  '''The parameters of ``SedScaledPosNegFocalLossAugd``.

  Attributes:
    gamma: The focal exponent.
    alpha_0: The scale of the negative positions.
    alpha_1: The scale of the positive positions.
    secondary_factor: The scale of the secondary species.
    primary_drop_out_factor: The probability that a primary species
      position is **dropped**.  The default of 0.3 means that 30 percent
      of the primary positions contribute nothing to the loss.
  '''

  gamma: float = 0.0
  alpha_0: float = 1.0
  alpha_1: float = 1.0
  secondary_factor: float = 0.1
  primary_drop_out_factor: float = 0.3


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class BCEControlledStandardLossParams:
  '''The parameters of ``BCEControlledStandardLoss``.

  Attributes:
    beta: The softness of the label.  The target becomes
      ``(beta * y_true + (1 - beta) * y_pred) * y_true``.
  '''

  beta: float = 0.5


CRITERION_PARAM_SCHEMAS: dict[str, Type] = {
    CRITERION_FOCAL_STANDARD: FocalLossStandardParams,
    CRITERION_REMOVED_FOCAL: SedRemovedFocalLossParams,
    CRITERION_SCALED_POS_NEG_FOCAL: SedScaledPosNegFocalLossParams,
    CRITERION_SCALED_POS_NEG_FOCAL_AUGD: SedScaledPosNegFocalLossAugdParams,
    CRITERION_BCE_CONTROLLED_STANDARD: BCEControlledStandardLossParams,
}
