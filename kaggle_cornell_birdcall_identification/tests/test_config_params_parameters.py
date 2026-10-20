#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the configuration contract.

The model is a pydantic **dataclass** rather than a ``BaseModel`` for
one reason: the original's configuration surface has an attribute
literally called ``model_config``, and a ``BaseModel`` reserves that
name for its own settings.

The three dataset parameter groups are **required**: a configuration
without data paths is not a recipe.
'''

from pathlib import Path

import pytest
from pydantic import ValidationError

from birdcall.config_params.parameters import METRIC_NAMES
from birdcall.config_params.parameters import Parameters
from birdcall.config_params.parameters import valid_criterion_names


def _dataset(root: str) -> dict:
  '''Builds the keys of one dataset parameter group.

  Args:
    root: The name of the split.

  Returns:
    A mapping of the required dataset keys.
  '''
  return {
      'root_dir': Path('/data'),
      'csv_dir': Path(f'/folds/{root}.csv'),
      'period': 30,
      'isTraining': root == 'train',
      'num_test_samples': 1,
  }


def _parameters(**overrides) -> Parameters:
  '''Builds a valid configuration.

  Args:
    **overrides: The values that differ from the default recipe.

  Returns:
    The validated configuration.
  '''
  values = {
      'train_ds_params': _dataset('train'),
      'valid_ds_params': _dataset('valid'),
      'test_ds_params': _dataset('test'),
  }
  values.update(overrides)
  return Parameters(**values)


def test_the_three_dataset_groups_are_required():
  '''A configuration without data paths is not a recipe.'''
  with pytest.raises(ValidationError):
    Parameters()


def test_the_defaults_are_the_originals():
  '''The default recipe is the one of the shipped experiments.'''
  parameters = _parameters()
  assert parameters.fold == 0
  assert parameters.name == 'birdcall'
  assert parameters.model_name == 'sed_dense121att'
  assert parameters.criterion_name == 'sed_scaled_pos_neg_focal_loss'
  assert parameters.optimizer_name == 'adamw'
  assert parameters.scheduler_name == 'warmup_with_cosine'
  assert parameters.aug_name == 'default'
  assert parameters.apply_mixup is False
  assert parameters.lr == 0.001
  assert parameters.weight_decay == 0.01
  assert parameters.lr_scale_factor == 0.01
  assert parameters.PERIOD == 30
  assert parameters.train_bs == 28
  assert parameters.valid_bs == 32
  assert parameters.eval_every == 2
  assert parameters.seed == 42
  assert parameters.mixup_seed == 1234
  assert parameters.deterministic is True
  assert parameters.load_model_only is True


def test_the_model_config_attribute_keeps_its_name():
  '''A ``BaseModel`` would reserve the name for its own settings.'''
  parameters = _parameters()
  assert parameters.model_config.sample_rate == 32000
  assert parameters.model_config.classes_num == 264


def test_the_criterion_parameters_default_to_the_inert_ones():
  '''With these parameters the focal term is inert.'''
  parameters = _parameters()
  assert parameters.criterion_params == {
      'gamma': 0.0,
      'alpha_0': 1.0,
      'alpha_1': 1.0,
      'secondary_factor': 1.0,
  }


def test_the_metrics_default_to_the_three_shipped_ones():
  '''The three validation metrics are the default set.'''
  parameters = _parameters()
  assert parameters.metrics == list(METRIC_NAMES)
  assert parameters.track_metric == 'f1score_clip'
  assert parameters.metric_factor == 1


def test_a_criterion_parameter_of_another_criterion_is_refused():
  '''The parameter set must match the selected criterion.'''
  with pytest.raises(ValueError):
    _parameters(criterion_name='focal_loss_standard',
                criterion_params={'gamma': 0.0, 'alpha': 1.0, 'beta': 0.5})


def test_a_tracked_metric_that_is_not_computed_is_refused():
  '''A checkpoint monitor that is never logged never fires.'''
  with pytest.raises(ValueError, match='track_metric'):
    _parameters(metrics=['lraps'], track_metric='f1score_clip')


def test_mixup_without_the_in_graph_augmentation_is_refused():
  '''The original halved the batch against unmixed data and crashed.'''
  model_config = {'sample_rate': 32000, 'window_size': 1024,
                  'hop_size': 320, 'mel_bins': 64, 'fmin': 50, 'fmax': 14000,
                  'classes_num': 264, 'apply_aug': False, 'top_db': None}
  with pytest.raises(ValueError, match='apply_mixup'):
    _parameters(apply_mixup=True, model_config=model_config)


def test_the_tracker_payload_is_derived_from_the_configuration():
  '''The payload cannot drift from the run it describes.'''
  parameters = _parameters()
  assert parameters.logger_params.params is not None
  assert parameters.logger_params.params['lr'] == 0.001
  assert parameters.logger_params.params['seed'] == 42
  assert parameters.logger_params.params['model_classes_num'] == 264
  assert parameters.logger_params.params['criterion_gamma'] == 0.0


def test_the_flat_parameters_carry_the_nested_ones():
  '''The nested configuration groups are flattened into the payload.'''
  parameters = _parameters()
  flat = parameters.flat_params()
  assert flat['max_epochs'] == parameters.run_params.max_epochs
  assert flat['n_saved'] == parameters.checkpoint_params.n_saved
  assert flat['model_window_size'] == 1024


def test_the_resolved_criterion_parameters_fill_in_the_defaults():
  '''The schema defaults reach the criterion constructor.'''
  parameters = _parameters(criterion_name='sed_scaled_pos_neg_focal_loss_augd',
                           criterion_params={'gamma': 0.0})
  resolved = parameters.resolved_criterion_params()
  assert resolved['gamma'] == 0.0
  assert resolved['alpha_0'] == 1.0
  assert resolved['alpha_1'] == 1.0
  assert resolved['secondary_factor'] == 0.1
  assert resolved['primary_drop_out_factor'] == 0.3


def test_the_reachable_criterion_names_are_published():
  '''The five criteria the SED path can select.'''
  assert valid_criterion_names() == (
      'focal_loss_standard',
      'sed_removed_focal_loss',
      'sed_scaled_pos_neg_focal_loss',
      'sed_scaled_pos_neg_focal_loss_augd',
      'bce_controlled_standard_loss',
  )


def test_an_unknown_key_is_refused():
  '''The configuration surface is closed.'''
  with pytest.raises(ValidationError):
    _parameters(not_a_key=1)
