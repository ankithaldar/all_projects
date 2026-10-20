#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The entry point of the PyTorch Lightning backend.

The command line keeps the original's shape: a configuration, a mode and
a couple of paths.  The configuration is a YAML file of the packaged
experiment set, resolved by ``birdcall.config_params.loader``, and the
backend selection says whether this backend runs at all.

The mode is one of:

``train``
    Fit the model, writing the checkpoints of the run.
``infer``
    Run the test split and write the score artefacts.
``vote``
    Hand over to the vote, which is framework agnostic and reads the
    artefacts the inference wrote.  The configuration of this mode is
    the **ensemble manifest** and the output directory holds the score
    artefacts; the submission is written next to them.
'''

import argparse
import logging
from pathlib import Path

from birdcall.callbacks.pl_callback_inference_dump import InferenceDumpCallback
from birdcall.callbacks.pl_callback_log_every import LogEveryN
from birdcall.callbacks.pl_callback_discord import DiscordCallback
from birdcall.config_params.backend import BackendSelection
from birdcall.config_params.loader import load_backend_selection
from birdcall.config_params.loader import load_parameters
from birdcall.config_params.parameters import Parameters
from birdcall.datamodules.pl_datamodule_sed import SedDataModule
from birdcall.engines.pl_lightning_module import SedLightningModule
from birdcall.engines.pl_trainer import build_model_checkpoint
from birdcall.engines.pl_trainer import build_trainer
from birdcall.engines.pl_trainer import load_weights_only
from birdcall.engines.pl_trainer import resume_checkpoint_path
from birdcall.helpers.pl_seed import seed_everything
from birdcall.loggers.pl_logger_print import PrintLoggerAdapter
from birdcall.provenance import write_run_manifest
from birdcall.loggers.pl_logger_wandb import WandbLoggerAdapter

LOGGER = logging.getLogger(__name__)

MODES = ('train', 'infer', 'vote')
BACKENDS = ('pl', 'tf', 'both')


def parse_args(argv=None) -> argparse.Namespace:
  '''Parses the command line.

  Args:
    argv: The arguments, or ``None`` for the ones of the process.

  Returns:
    The parsed arguments.
  '''
  parser = argparse.ArgumentParser(
      description='Train, infer or vote with the PyTorch Lightning backend.')
  parser.add_argument('--config', required=True,
                      help='The path of the experiment YAML file, or of the '
                           'ensemble manifest in the vote mode.')
  parser.add_argument('--backend', choices=BACKENDS, default=None,
                      help='Which backend runs. It overrides the packaged '
                           'config_params/run/backend.yaml.')
  parser.add_argument('--mode', choices=MODES, default='train',
                      help='What the run does.')
  parser.add_argument('--checkpoint', default=None,
                      help='The checkpoint the run loads.')
  parser.add_argument('--output', default=None,
                      help='The directory the inference artefacts are '
                           'written to.')
  return parser.parse_args(argv)


def resolve_backend(backend: str | None = None) -> BackendSelection:
  '''Reads the backend selection.

  Args:
    backend: The selection of the command line, or ``None`` to use the
      packaged one.

  Returns:
    The selection, with the command line winning over the packaged file.
  '''
  document = load_backend_selection()
  selection = BackendSelection(**document)
  if backend is not None:
    selection.backend = backend
  return selection


def resolve_configuration(config_path) -> Parameters:
  '''Loads and validates the configuration of a run.

  Args:
    config_path: The path of the experiment YAML file.

  Returns:
    The validated configuration.

  Raises:
    FileNotFoundError: If the file does not exist.
  '''
  return load_parameters(config_path)


def build_logger(parameters: Parameters):
  '''Builds the logger of a run.

  Args:
    parameters: The validated configuration.

  Returns:
    The logger of the configured tracker.
  '''
  if parameters.logger_name == 'print':
    return PrintLoggerAdapter(parameters)
  return WandbLoggerAdapter(parameters)


def build_callbacks(parameters: Parameters) -> list:
  '''Builds the callbacks of a run.

  Args:
    parameters: The validated configuration.

  Returns:
    The checkpoint callback, the every-N-steps logger and the Discord
    notifier.
  '''
  return [
      build_model_checkpoint(parameters),
      LogEveryN(parameters.logger_params.log_every),
      DiscordCallback(),
  ]


def run_training(parameters: Parameters, module: SedLightningModule,
                 datamodule: SedDataModule, trainer) -> None:
  '''Fits the model.

  Args:
    parameters: The validated configuration.
    module: The module of the run.
    datamodule: The datamodule of the run.
    trainer: The trainer of the run.
  '''
  resume = resume_checkpoint_path(parameters)
  if parameters.load_model_only:
    if resume is not None:
      load_weights_only(module, resume)
    trainer.fit(module, datamodule)
    return
  trainer.fit(module, datamodule, ckpt_path=str(resume) if resume else None)


def run_inference(parameters: Parameters, module: SedLightningModule,
                  datamodule: SedDataModule, trainer) -> None:
  '''Runs the test split and writes the score artefacts.

  Args:
    parameters: The validated configuration.
    module: The module of the run.
    datamodule: The datamodule of the run.
    trainer: The trainer of the run.
  '''
  resume = resume_checkpoint_path(parameters)
  if resume is not None:
    load_weights_only(module, resume)
  trainer.test(module, datamodule)


def run_vote(args: argparse.Namespace) -> int:
  '''Hands over to the vote.

  The vote is framework agnostic: it reads the score artefacts the
  inference wrote, one per model and per audio file.  Its own entry
  point takes the ensemble manifest, the directory of the artefacts and
  the path of the submission, which is what the three flags of this
  entry point map onto.

  Args:
    args: The parsed arguments of the entry point.

  Returns:
    The exit code of the vote.

  Raises:
    RuntimeError: If the vote entry point is not installed.
  '''
  try:
    from birdcall.run_vote import main as vote_main
  except ImportError as error:
    raise RuntimeError(
        'The vote entry point is not available. The vote is framework '
        'agnostic and reads the artefacts the inference wrote.') from error
  scores = Path(args.output) if args.output else Path('scores')
  vote_argv = [
      '--ensemble', str(args.config),
      '--scores', str(scores),
      '--output', str(scores / 'submission.csv'),
  ]
  return vote_main(vote_argv)


def main(argv=None) -> int:
  '''Runs the backend.

  Args:
    argv: The arguments, or ``None`` for the ones of the process.

  Returns:
    The exit code of the run.
  '''
  args = parse_args(argv)
  selection = resolve_backend(args.backend)
  if not selection.is_enabled('pl'):
    LOGGER.info('The PyTorch Lightning backend is not selected; '
                'nothing to do.')
    return 0
  if args.mode == 'vote':
    return run_vote(args)
  parameters = resolve_configuration(args.config)
  if args.checkpoint is not None:
    parameters.checkpoint_dir = Path(args.checkpoint)
  seed_everything(parameters.seed, deterministic=parameters.deterministic)
  module = SedLightningModule(parameters)
  datamodule = SedDataModule(parameters)
  callbacks = build_callbacks(parameters)
  if args.mode == 'infer':
    output = (Path(args.output) if args.output
              else parameters.checkpoint_params.save_dir)
    callbacks.append(InferenceDumpCallback(output))
  trainer = build_trainer(parameters,
                          callbacks=callbacks,
                          logger=build_logger(parameters))
  if args.mode == 'train':
    run_training(parameters, module, datamodule, trainer)
    write_run_manifest(parameters, args.config, mode=args.mode)
    return 0
  run_inference(parameters, module, datamodule, trainer)
  write_run_manifest(parameters, args.config, mode=args.mode)
  return 0
