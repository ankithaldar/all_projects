#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The configuration contract: one pydantic model, validated.

Every experiment is one YAML file, and this module is the only place a
configuration attribute is named.  The model is a pydantic **dataclass**
rather than a ``BaseModel`` for one reason: the original's configuration
surface has an attribute literally called ``model_config``, and a
``BaseModel`` reserves that name for its own settings.

The three dataset parameter groups are **required**: a configuration without
data paths is not a recipe.  Everything else has a default, so a config
file only has to state what makes it different.
'''

from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional

from pydantic import ConfigDict
from pydantic import ValidationError
from pydantic.dataclasses import dataclass

from birdcall.config_params.checkpoint_params import CheckpointParams
from birdcall.config_params.criterion_params import CRITERION_NAMES
from birdcall.config_params.criterion_params import CRITERION_PARAM_SCHEMAS
from birdcall.config_params.dataset_params import DatasetParams
from birdcall.config_params.logger_params import LoggerParams
from birdcall.config_params.model_config import ModelConfig
from birdcall.config_params.run_params import RunParams

ModelName = Literal['sed_dense121att']
CriterionName = Literal[
    'focal_loss_standard',
    'sed_removed_focal_loss',
    'sed_scaled_pos_neg_focal_loss',
    'sed_scaled_pos_neg_focal_loss_augd',
    'bce_controlled_standard_loss',
]
SchedulerName = Literal['none', 'warmup_with_cosine', 'one_cycle_cosine']
OptimizerName = Literal['adamw']
LoggerName = Literal['wandb', 'print']
AugmentationName = Literal['default', 'secondary_default']
TrackMetric = Literal['lraps', 'f1score_clip', 'f1score_frame']

METRIC_NAMES = ('lraps', 'f1score_clip', 'f1score_frame')


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class Parameters:
  '''The validated configuration of one experiment.

  Attributes:
    train_ds_params: The training split.
    valid_ds_params: The validation split.
    test_ds_params: The test split.
    fold: The fold number, or ``'all'`` for a whole-data retrain.
    name: The name of the experiment.
    model_name: The model variant.
    model_config: The model hyper-parameters.
    pretrained_path: A checkpoint to warm start from.  Only the
      ``sed_ccnn14att`` branch of the original honoured it.
    aug_name: The waveform augmentation stack.
    apply_mixup: Whether mixup runs.
    bckgrd_aug_dir: The directory of background noise.
    secondary_bckgrd_aug_dir: The directory of short background noises.
    criterion_name: The criterion.
    criterion_params: The criterion hyper-parameters.
    optimizer_name: The optimiser.
    weight_decay: The weight decay.
    lr: The base learning rate.
    scheduler_name: The learning rate schedule.
    lr_scale_factor: The factor the learning rate is scaled by.
    PERIOD: The snippet length in seconds.
    train_bs: The training batch size.
    train_num_workers: The number of training dataloader workers.
    valid_bs: The validation batch size.
    valid_num_workers: The number of validation dataloader workers.
    metrics: The metrics that are computed.
    track_metric: The metric the checkpoints are ranked by.
    metric_factor: The sign of the metric: 1 to maximise, -1 to minimise.
    logger_name: The tracker.
    logger_params: The tracker parameters.
    checkpoint_params: The checkpointing parameters.
    checkpoint_dir: The directory a checkpoint is resumed from.
    load_model_only: Whether the resume restores only the weights.  True in
      every shipped config, and it means the optimiser is **not**
      restored.
    run_params: The training loop parameters.
    dist_params: The distributed parameters.
    val_length: The number of validation iterations, or ``None``.
    eval_every: The validation cadence in epochs.
    add_pbar: Whether a progress bar is shown.
    accumulation_steps: The gradient accumulation factor.
    gradient_clip_val: The gradient clipping value.  Zero disables it.
    seed: The root seed of the experiment.
    mixup_seed: The seed of the mixup coefficient sequence.  The original
      hard coded 1234 in a default argument, where nobody could see it.
    deterministic: Whether deterministic kernels are requested.
  '''

  train_ds_params: DatasetParams
  valid_ds_params: DatasetParams
  test_ds_params: DatasetParams
  fold: int | str = 0
  name: str = 'birdcall'
  model_name: ModelName = 'sed_dense121att'
  model_config: ModelConfig = field(default_factory=ModelConfig)
  pretrained_path: Optional[Path] = None
  aug_name: AugmentationName = 'default'
  apply_mixup: bool = False
  bckgrd_aug_dir: Optional[Path] = None
  secondary_bckgrd_aug_dir: Optional[Path] = None
  criterion_name: CriterionName = 'sed_scaled_pos_neg_focal_loss'
  criterion_params: dict[str, float] = field(
      default_factory=lambda: {
          'gamma': 0.0,
          'alpha_0': 1.0,
          'alpha_1': 1.0,
          'secondary_factor': 1.0,
      })
  optimizer_name: OptimizerName = 'adamw'
  weight_decay: float = 0.01
  lr: float = 0.001
  scheduler_name: SchedulerName = 'warmup_with_cosine'
  lr_scale_factor: float = 0.01
  PERIOD: int = 30
  train_bs: int = 28
  train_num_workers: int = 8
  valid_bs: int = 32
  valid_num_workers: int = 4
  metrics: list[str] = field(
      default_factory=lambda: list(METRIC_NAMES))
  track_metric: TrackMetric = 'f1score_clip'
  metric_factor: int = 1
  logger_name: LoggerName = 'wandb'
  logger_params: LoggerParams = field(default_factory=LoggerParams)
  checkpoint_params: CheckpointParams = field(
      default_factory=lambda: CheckpointParams(save_dir=Path('saved_models')))
  checkpoint_dir: Optional[Path] = None
  load_model_only: bool = True
  run_params: RunParams = field(default_factory=RunParams)
  dist_params: dict[str, Any] = field(default_factory=dict)
  val_length: Optional[int] = None
  eval_every: int = 2
  add_pbar: bool = True
  accumulation_steps: int = 1
  gradient_clip_val: float = 0.0
  seed: int = 42
  mixup_seed: int = 1234
  deterministic: bool = True

  def __post_init__(self) -> None:
    '''Validates the cross-field constraints of the configuration.

    Raises:
      ValueError: If the configuration is internally inconsistent.
    '''
    self._validate_criterion_params()
    self._validate_track_metric()
    self._validate_augmentation()
    self._derive_logger_params()

  def _validate_criterion_params(self) -> None:
    '''Checks the criterion parameters against the selected criterion.

    Raises:
      ValueError: If a parameter does not belong to the criterion.
    '''
    schema = CRITERION_PARAM_SCHEMAS[self.criterion_name]
    try:
      schema(**self.criterion_params)
    except TypeError as error:
      raise ValueError(
          f'criterion_params {sorted(self.criterion_params)} do not match '
          f'criterion_name {self.criterion_name!r}: {error}') from error

  def _validate_track_metric(self) -> None:
    '''Checks that the tracked metric is one of the computed metrics.

    Raises:
      ValueError: If the tracked metric is not computed.
    '''
    if self.track_metric not in self.metrics:
      raise ValueError(
          f'track_metric {self.track_metric!r} is not in metrics '
          f'{self.metrics}; a checkpoint monitor that is never logged '
          'silently never fires.')

  def _validate_augmentation(self) -> None:
    '''Checks that the augmentation combination is coherent.

    Raises:
      ValueError: If mixup is on while the in-graph SpecAugment is off.
    '''
    if self.apply_mixup and not self.model_config.apply_aug:
      raise ValueError(
          'apply_mixup is on while model_config.apply_aug is off. The '
          'original halved the batch against unmixed data in that '
          'combination and crashed; fix the configuration instead.')

  def _derive_logger_params(self) -> None:
    '''Fills the tracker hyper-parameter payload from this configuration.'''
    if self.logger_params.params is None:
      self.logger_params.params = self.flat_params()

  def flat_params(self) -> dict[str, Any]:
    '''Flattens the configuration into a tracker friendly dictionary.

    Returns:
      A flat mapping of every configuration value that is not a path or a
      mapping of paths.
    '''
    payload: dict[str, Any] = {
        'name': self.name,
        'fold': self.fold,
        'model_name': self.model_name,
        'criterion_name': self.criterion_name,
        'optimizer_name': self.optimizer_name,
        'scheduler_name': self.scheduler_name,
        'aug_name': self.aug_name,
        'apply_mixup': self.apply_mixup,
        'weight_decay': self.weight_decay,
        'lr': self.lr,
        'lr_scale_factor': self.lr_scale_factor,
        'period': self.PERIOD,
        'train_bs': self.train_bs,
        'valid_bs': self.valid_bs,
        'eval_every': self.eval_every,
        'accumulation_steps': self.accumulation_steps,
        'gradient_clip_val': self.gradient_clip_val,
        'track_metric': self.track_metric,
        'metric_factor': self.metric_factor,
        'max_epochs': self.run_params.max_epochs,
        'epoch_length': self.run_params.epoch_length,
        'load_model_only': self.load_model_only,
        'n_saved': self.checkpoint_params.n_saved,
        'seed': self.seed,
        'mixup_seed': self.mixup_seed,
        'deterministic': self.deterministic,
    }
    payload.update({f'model_{key}': value
                    for key, value in vars(self.model_config).items()})
    payload.update({f'criterion_{key}': value
                    for key, value in self.criterion_params.items()})
    return payload

  def resolved_criterion_params(self) -> dict[str, float]:
    '''Returns the criterion parameters with the schema defaults filled in.

    Returns:
      The parameters the criterion constructor will receive.
    '''
    schema = CRITERION_PARAM_SCHEMAS[self.criterion_name]
    instance = schema(**self.criterion_params)
    return dict(vars(instance))


def valid_criterion_names() -> tuple[str, ...]:
  '''Returns the names of the reachable criteria.

  Returns:
    The five criteria the SED path can select.
  '''
  return CRITERION_NAMES


__all__ = [
    'METRIC_NAMES',
    'Parameters',
    'ValidationError',
    'valid_criterion_names',
]
