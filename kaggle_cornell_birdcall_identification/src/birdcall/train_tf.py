#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The entry point of the Keras backend.

The command line is the original's, with the arguments the port needs:
``--config`` names the experiment, ``--backend`` says which backend
runs, ``--mode`` says what the run does, ``--checkpoint`` is the
checkpoint to resume from, and ``--output`` is where the run writes.

The ``--config`` value is the path of an experiment YAML file, or the
dotted path of a module holding a ``Parameters`` class, so both of the
project's configuration conventions work.

The training loop is Keras': ``compile`` and ``fit``.  What the port
keeps from the original is everything the framework does not replace:
the mixup halving, the loss components, the checkpoint ranking and the
logged series.
'''

import argparse
import importlib
import logging
import sys
from pathlib import Path

from birdcall.augmentations.registry import build_transforms
from birdcall.callbacks.tf_callback_keep_best_n import KeepBestNCallback
from birdcall.callbacks.tf_callback_discord import DiscordCallback
from birdcall.config_params.loader import load_parameters
from birdcall.config_params.parameters import Parameters
from birdcall.datasets.tf_dataset_sed import SedDataset
from birdcall.engines.tf_keras_model import SedKerasModel
from birdcall.helpers.tf_seed import seed_everything
from birdcall.loggers.tf_logger_print import PrintLogger
from birdcall.loggers.tf_logger_wandb import WandbLogger
from birdcall.losses.tf_registry import build_criterion
from birdcall.metrics.tf_registry import build_metric
from birdcall.provenance import write_run_manifest
from birdcall.models.tf_registry import build_model
from birdcall.optimizers.tf_optimizer_adamw import build_optimizer
from birdcall.optimizers.tf_optimizer_adamw import build_scheduler

LOGGER = logging.getLogger(__name__)

BACKENDS = ('pl', 'tf', 'both')
MODES = ('train', 'valid', 'test')


def build_parser() -> argparse.ArgumentParser:
  '''Builds the command line parser.

  Returns:
    The parser.
  '''
  parser = argparse.ArgumentParser(
      description='Train the Cornell Birdcall model with TensorFlow/Keras.')
  parser.add_argument('--config', required=True,
                      help='The experiment YAML file, or the dotted path '
                           'of a module holding a Parameters class.')
  parser.add_argument('--backend', default='tf', choices=BACKENDS,
                      help='Which backend runs.')
  parser.add_argument('--mode', default='train', choices=MODES,
                      help='What the run does.')
  parser.add_argument('--checkpoint', default=None,
                      help='The checkpoint to resume from.')
  parser.add_argument('--output', default=None,
                      help='The directory the run writes to.')
  return parser


def load_config(value: str) -> Parameters:
  '''Loads a configuration from a path or from a dotted module path.

  Args:
    value: The path of an experiment YAML file, or the dotted path of a
      module holding a ``Parameters`` class.

  Returns:
    The validated configuration.

  Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the module holds no ``Parameters`` class.
  '''
  path = Path(value)
  if path.suffix in ('.yaml', '.yml') or path.exists():
    return load_parameters(path)
  module = importlib.import_module(value)
  parameters = getattr(module, 'Parameters', None)
  if parameters is None:
    raise ValueError(f'{value} holds no Parameters class.')
  return parameters()


def build_datasets(config: Parameters, transforms: dict):
  '''Builds the training and the validation datasets.

  Args:
    config: The validated configuration.
    transforms: The waveform augmentation stack.

  Returns:
    A tuple ``(train dataset, validation dataset)``.
  '''
  train = SedDataset(
      root_dir=config.train_ds_params.root_dir,
      csv_dir=config.train_ds_params.csv_dir,
      period=config.train_ds_params.period,
      bird_code=config.train_ds_params.bird_code,
      inv_ebird_label=config.train_ds_params.inv_ebird_label,
      isTraining=config.train_ds_params.isTraining,
      num_test_samples=config.train_ds_params.num_test_samples,
      transform=transforms['train'],
      background_audio_dir=config.train_ds_params.background_audio_dir,
      seed=config.seed)
  valid = SedDataset(
      root_dir=config.valid_ds_params.root_dir,
      csv_dir=config.valid_ds_params.csv_dir,
      period=config.valid_ds_params.period,
      bird_code=config.valid_ds_params.bird_code,
      inv_ebird_label=config.valid_ds_params.inv_ebird_label,
      isTraining=config.valid_ds_params.isTraining,
      num_test_samples=config.valid_ds_params.num_test_samples,
      transform=transforms['valid'],
      background_audio_dir=config.valid_ds_params.background_audio_dir,
      seed=config.seed)
  return train, valid


def build_logger(config: Parameters):
  '''Builds the tracker of the run.

  Args:
    config: The validated configuration.

  Returns:
    The tracker callback.
  '''
  params = config.logger_params
  if config.logger_name == 'print':
    return PrintLogger(log_every=params.log_every)
  return WandbLogger(project_name=params.project_name, name=params.name,
                     prefix_name=params.prefix_name, tags=params.tags,
                     params=params.params, log_every=params.log_every)


def build_callbacks(config: Parameters, output_dir: Path) -> list:
  '''Builds the callback list of the run.

  The order matters: the tracker sees the epoch metrics before the
  checkpoint callback ranks them, and the Discord message is the last
  thing that happens in an epoch.

  Args:
    config: The validated configuration.
    output_dir: The directory the run writes to.

  Returns:
    The list of callbacks.
  '''
  callbacks = [build_logger(config)]
  callbacks.append(KeepBestNCallback(
      save_dir=output_dir,
      n_saved=config.checkpoint_params.n_saved,
      monitor=config.track_metric,
      mode='max' if config.metric_factor >= 0 else 'min',
      prefix_name=config.checkpoint_params.prefix_name))
  callbacks.append(DiscordCallback())
  return callbacks


def run(config: Parameters, mode: str = 'train', checkpoint=None,
        output_dir: Path | None = None) -> int:
  '''Runs one experiment.

  Args:
    config: The validated configuration.
    mode: What the run does.
    checkpoint: The checkpoint to resume from.
    output_dir: The directory the run writes to.

  Returns:
    The exit code of the run.
  '''
  seed_everything(config.seed, deterministic=config.deterministic)
  try:
    transforms = build_transforms(
        config.aug_name, bckgrd_aug_dir=config.bckgrd_aug_dir,
        secondary_bckgrd_aug_dir=config.secondary_bckgrd_aug_dir)
  except ImportError as error:
    # audiomentations is optional: it pins an old librosa, which
    # conflicts with the mel matrix build.  A run without it has no
    # waveform augmentation, and says so.
    LOGGER.warning('The waveform augmentation is not available: %s', error)
    transforms = {'train': None, 'valid': None}
  train_dataset, valid_dataset = build_datasets(config, transforms)
  output_dir = Path(output_dir or config.checkpoint_params.save_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  backbone = build_model(config.model_name, **vars(config.model_config))
  criterion = build_criterion(config.criterion_name,
                               **config.resolved_criterion_params())
  model = SedKerasModel(backbone, criterion=criterion,
                         apply_mixup=config.apply_mixup,
                         mixup_seed=config.mixup_seed)
  if checkpoint is not None:
    model.load_weights(checkpoint)

  train_data = train_dataset.build(batch_size=config.train_bs)
  steps_per_epoch = (config.run_params.epoch_length or
                      train_dataset.cardinality())
  optimizer = build_optimizer({
      'lr': config.lr,
      'weight_decay': config.weight_decay,
      'clipnorm': config.gradient_clip_val or None,
      'gradient_accumulation_steps': config.accumulation_steps,
  }, model)
  optimizer.learning_rate = build_scheduler({
      'scheduler_name': config.scheduler_name,
      'lr': config.lr,
      'lr_scale_factor': config.lr_scale_factor,
      'num_epochs': config.run_params.max_epochs,
  }, steps_per_epoch)
  metrics = [build_metric(name) for name in config.metrics]
  for metric in metrics:
    model.add_validation_metric(metric.name, metric)
  model.compile(optimizer=optimizer, loss=criterion)
  callbacks = build_callbacks(config, output_dir)
  if mode == 'train':
    model.fit(train_data, epochs=config.run_params.max_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset.build(
                  batch_size=config.valid_bs),
              validation_freq=config.eval_every,
              callbacks=callbacks)
  else:
    model.evaluate(valid_dataset.build(batch_size=config.valid_bs),
                   callbacks=callbacks)
  return 0


def main(argv=None) -> int:
  '''The console entry point.

  Args:
    argv: The command line arguments.

  Returns:
    The exit code of the run.
  '''
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(levelname)s %(name)s: %(message)s')
  args = build_parser().parse_args(argv)
  if args.backend not in ('tf', 'both'):
    LOGGER.info('The %s backend is not run by this entry point.', args.backend)
    return 0
  config = load_config(args.config)
  output = (Path(args.output) if args.output else
            config.checkpoint_params.save_dir)
  try:
    result = run(config, mode=args.mode, checkpoint=args.checkpoint,
                 output_dir=output)
    write_run_manifest(config, Path(args.config), mode=args.mode)
    return result
  except Exception:  # pylint: disable=broad-except
    LOGGER.exception('The run failed.')
    return 1


if __name__ == '__main__':
  sys.exit(main())
