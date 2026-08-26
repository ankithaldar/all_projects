#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Weights & Biases logger construction from configuration.

The API key is resolved through ``knee.helpers.secrets`` (environment ->
``.env`` -> Kaggle secrets). When no key is available the logger degrades to
offline mode so runs remain resumable and never block training.
"""

from __future__ import annotations

import os

from knee.helpers.secrets import get_secret
from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

DEFAULT_MODE = 'offline'
ONLINE_MODE = 'online'


def build_wandb_logger(config: dict, fold_id: int):
  """Create a Lightning WandbLogger for one fold.

  Args:
      config: Composed experiment configuration; uses
          ``integrations.wandb`` (``enabled``, ``project``, ``entity``,
          ``api_key_secret``) plus ``experiment.name`` and
          ``experiment.output_dir``.
      fold_id: Fold appended to the run name.

  Returns:
      ``pytorch_lightning.loggers.WandbLogger`` or None when disabled.
  """
  # Imported lazily so `main.py infer` never pays the wandb import cost.
  # pylint: disable=import-outside-toplevel
  from pytorch_lightning.loggers import WandbLogger
  # pylint: enable=import-outside-toplevel

  wandb_cfg = config.get('integrations', {}).get('wandb', {})
  if not wandb_cfg.get('enabled'):
    return None

  api_key = get_secret(wandb_cfg.get('api_key_secret', 'WANDB_API_KEY'))
  mode = wandb_cfg.get('mode') or (ONLINE_MODE if api_key else DEFAULT_MODE)
  if api_key:
    os.environ['WANDB_API_KEY'] = api_key
  else:
    _LOGGER.warning(
      'No W&B API key resolved (env/.env/Kaggle secrets); '
      'falling back to mode=%s',
      mode,
    )

  experiment_name = config['experiment']['name']
  return WandbLogger(
    project=wandb_cfg.get('project', 'knee-mvp'),
    entity=wandb_cfg.get('entity'),
    name=f'{experiment_name}-fold{fold_id}',
    save_dir=config['experiment']['output_dir'],
    mode=mode,
  )
