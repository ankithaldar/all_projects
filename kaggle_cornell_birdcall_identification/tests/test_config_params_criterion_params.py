#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the per-criterion hyper-parameter schemas.

Each of the five reachable criteria takes a different set of parameters.
The original passed a plain dict straight into the criterion
constructor, so a parameter meant for one criterion was silently
accepted by another.
'''

import pytest
from pydantic import ValidationError

from birdcall.config_params.criterion_params import (
    CRITERION_BCE_CONTROLLED_STANDARD,
)
from birdcall.config_params.criterion_params import CRITERION_FOCAL_STANDARD
from birdcall.config_params.criterion_params import CRITERION_NAMES
from birdcall.config_params.criterion_params import CRITERION_PARAM_SCHEMAS
from birdcall.config_params.criterion_params import CRITERION_REMOVED_FOCAL
from birdcall.config_params.criterion_params import (
    CRITERION_SCALED_POS_NEG_FOCAL,
)
from birdcall.config_params.criterion_params import (
    CRITERION_SCALED_POS_NEG_FOCAL_AUGD,
)
from birdcall.config_params.criterion_params import (
    BCEControlledStandardLossParams,
)
from birdcall.config_params.criterion_params import FocalLossStandardParams
from birdcall.config_params.criterion_params import SedRemovedFocalLossParams
from birdcall.config_params.criterion_params import (
    SedScaledPosNegFocalLossAugdParams,
)
from birdcall.config_params.criterion_params import (
    SedScaledPosNegFocalLossParams,
)


def test_the_five_reachable_criteria_are_named():
  '''The names are the ones the configs use.'''
  assert CRITERION_NAMES == (
      'focal_loss_standard',
      'sed_removed_focal_loss',
      'sed_scaled_pos_neg_focal_loss',
      'sed_scaled_pos_neg_focal_loss_augd',
      'bce_controlled_standard_loss',
  )


def test_every_criterion_has_a_schema():
  '''The five names and the five schemas agree.'''
  assert set(CRITERION_PARAM_SCHEMAS) == set(CRITERION_NAMES)
  assert len(CRITERION_PARAM_SCHEMAS) == 5


def test_the_schemas_are_mapped_to_the_right_criteria():
  '''Each name points at the schema of its own criterion.'''
  assert (CRITERION_PARAM_SCHEMAS[CRITERION_FOCAL_STANDARD]
          is FocalLossStandardParams)
  assert (CRITERION_PARAM_SCHEMAS[CRITERION_REMOVED_FOCAL]
          is SedRemovedFocalLossParams)
  assert (CRITERION_PARAM_SCHEMAS[CRITERION_SCALED_POS_NEG_FOCAL]
          is SedScaledPosNegFocalLossParams)
  assert (CRITERION_PARAM_SCHEMAS[CRITERION_SCALED_POS_NEG_FOCAL_AUGD]
          is SedScaledPosNegFocalLossAugdParams)
  assert (CRITERION_PARAM_SCHEMAS[CRITERION_BCE_CONTROLLED_STANDARD]
          is BCEControlledStandardLossParams)


def test_the_focal_schema_defaults_to_the_inert_parameters():
  '''With `gamma = 0` the focal term is inert.'''
  params = FocalLossStandardParams()
  assert params.gamma == 0.0
  assert params.alpha == 1.0


def test_the_scaled_schema_defaults_to_the_originals():
  '''The secondary factor of the original is 0.1.'''
  params = SedScaledPosNegFocalLossParams()
  assert params.gamma == 0.0
  assert params.alpha_0 == 1.0
  assert params.alpha_1 == 1.0
  assert params.secondary_factor == 0.1


def test_the_augmented_schema_adds_the_dropout_factor():
  '''The dropout keeps thirty percent of the primary positions.'''
  params = SedScaledPosNegFocalLossAugdParams()
  assert params.primary_drop_out_factor == 0.3


def test_the_bce_schema_only_knows_beta():
  '''The L-soft BCE takes one parameter.'''
  params = BCEControlledStandardLossParams()
  assert params.beta == 0.5
  with pytest.raises(ValidationError):
    BCEControlledStandardLossParams(gamma=1.0)


def test_a_parameter_of_another_criterion_is_refused():
  '''The schemas are closed, so a stray parameter is a loud failure.'''
  with pytest.raises(ValidationError):
    SedScaledPosNegFocalLossParams(beta=0.5)
  with pytest.raises(ValidationError):
    FocalLossStandardParams(secondary_factor=0.1)
