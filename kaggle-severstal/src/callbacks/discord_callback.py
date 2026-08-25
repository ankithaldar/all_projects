#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Discord Logger to get realtime training updates'''


# imports
import os
import time
import random
from datetime import datetime
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import requests
from pytorch_lightning.utilities import rank_zero_only

#    script imports
# imports


# constants
USER_AGENT_LIST = [
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.5 Safari/605.1.15',
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.53 Safari/537.36',
  'Mozilla/5.0 (Windows NT 10.0; Windows; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36',
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/603.3.8 (KHTML, like Gecko) Version/10.1.2 Safari/603.3.8',
  'Mozilla/5.0 (Windows NT 10.0; Windows; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36',
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.53 Safari/537.36',
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
  'Mozilla/5.0 (Windows NT 10.0; Windows; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36',
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.53 Safari/537.36'
]
# constants


# classes
class DiscordCallback(pl.Callback):
  '''Discord Logger to get realtime training updates'''

  def __init__(
    self,
    webhook_url: str = None,
    webhook_env_var: str = 'DISCORD_WEBHOOK_URL',
    experiment_name: str = 'Untitled Experiment',
    log_every_n_steps: int = 100,
    log_val_metrics: bool = True,
    log_model_hyperparams: bool = True,
    max_retries: int = 3,
  ):
    '''
    Args:
      webhook_url: Discord webhook URL (either provide this or set webhook_env_var)
      webhook_env_var: Environment variable name containing webhook URL
      log_every_n_steps: Log training metrics every N steps
      log_val_metrics: Whether to log validation metrics
      log_model_hyperparams: Whether to log model hyperparameters
      max_retries: Maximum number of retries for failed webhook requests
    '''
    super().__init__()
    self.webhook_url = os.getenv(webhook_env_var).strip().strip('"').strip("'") or webhook_url

    if not self.webhook_url:
      raise ValueError(
        f'Discord webhook URL not provided and environment variable '
        f'"{webhook_env_var}" not found. Either pass webhook_url directly '
        f'or set the {webhook_env_var} environment variable.'
      )

    self.experiment_name = experiment_name
    self.log_every_n_steps = log_every_n_steps
    self.log_val_metrics = log_val_metrics
    self.log_model_hyperparams = log_model_hyperparams
    self.max_retries = max_retries
    self.start_time = datetime.now()

  @rank_zero_only
  def _send_to_discord(self, content: str, embed: Optional[Dict[str, Any]] = None):
    '''Helper method to send messages to Discord with error handling'''
    data = {'content': content}
    if embed:
      data['embeds'] = [embed]

    for attempt in range(self.max_retries):
      try:
        response = requests.post(
          self.webhook_url,
          json=data,
          headers={'Content-Type': 'application/json', 'User-Agent': USER_AGENT_LIST[random.randint(0, len(USER_AGENT_LIST)-1)]},
          timeout=5  # seconds
        )
        if response.status_code == 204:
          return
        elif response.status_code == 429:
          # Rate limited - wait and retry
          retry_after = float(response.json().get('retry_after', 1.0))
          time.sleep(retry_after)
          continue
        else:
          print(f'Discord webhook error (attempt {attempt + 1}/{self.max_retries}): {response.status_code} - {response.text}')
      except Exception as e:
        print(f'Discord webhook exception (attempt {attempt + 1}/{self.max_retries}): {str(e)}')

      if attempt < self.max_retries - 1:
        time.sleep(2 ** attempt)  # Exponential backoff

    print('Failed to send message to Discord after multiple attempts')

  @rank_zero_only
  def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    '''Log experiment start and hyperparameters'''
    # Create a formatted message with experiment info
    start_msg = (
      '--------------------------------------------------------------------------------------------------------------\n'
      f'**Training Started**: {self.experiment_name}'
    )

    embed = {
      'title': 'Experiment Details',
      'color': 0x3498db,  # Blue color
      'fields': [],
      'timestamp': datetime.utcnow().isoformat(),
    }

    # Add basic info
    embed['fields'].extend([
      {'name': 'Start Time', 'value': self.start_time.strftime('%Y-%m-%d %H:%M:%S'), 'inline': True},
      {'name': 'Device', 'value': str(trainer.strategy.root_device), 'inline': True},
      {'name': 'Precision', 'value': str(trainer.precision), 'inline': True},
    ])

    # Add hyperparameters if requested
    if self.log_model_hyperparams and hasattr(pl_module, 'hparams'):
      hparams = pl_module.hparams
      if isinstance(hparams, dict):
        hparams_str = '\n'.join(f'{k}: {v}' for k, v in hparams.items())
      else:
        hparams_str = str(hparams)

      embed['fields'].append({
        'name': 'Hyperparameters',
        'value': f'```\n{hparams_str[:1000]}\n```',  # Limit to 1000 chars
        'inline': False
      })

    self._send_to_discord(start_msg, embed)

  @rank_zero_only
  def on_train_batch_end(
    self,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    outputs: Dict[str, Any],
    batch: Any,
    batch_idx: int,
  ):
    '''Log training metrics periodically'''
    if (batch_idx + 1) % self.log_every_n_steps != 0:
      return

    metrics = trainer.callback_metrics
    current_epoch = trainer.current_epoch
    global_step = trainer.global_step

    # Format the message
    message = (
      f'**Training Update** - Epoch {current_epoch}, Batch {batch_idx} - *{datetime.utcnow().isoformat()}*\n'
      f'Step: {global_step} | '
      f'Loss: {metrics.get("train_loss", "N/A"):.4f}'
    )

    # Add additional metrics if available
    additional_metrics = []
    for k, v in metrics.items():
      if k.startswith('train_') and k != 'train_loss':
        additional_metrics.append(f'{k}: {v:.4f}')

    if additional_metrics:
      message += ' | ' + ' | '.join(additional_metrics)

    self._send_to_discord(message)

  @rank_zero_only
  def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    '''Log validation metrics at the end of validation'''
    if not self.log_val_metrics:
      return

    metrics = trainer.callback_metrics
    current_epoch = trainer.current_epoch

    # Filter validation metrics
    val_metrics = {k: v for k, v in metrics.items() if k.startswith('val_')}
    if not val_metrics:
      return

    # Format the message
    message = (
      f'**Validation Results** - Epoch {current_epoch} - *{datetime.utcnow().isoformat()}*\n'
      + '\n'.join(f'{k}: {v:.4f}' for k, v in val_metrics.items())
    )

    self._send_to_discord(message)

  @rank_zero_only
  def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    'Log training completion'
    duration = datetime.now() - self.start_time
    message = (
      f'**Training Complete**: {self.experiment_name} - *{datetime.utcnow().isoformat()}*\n'
      f'Total duration: {str(duration)}\n'
      f'Final epoch: {trainer.current_epoch}\n'
      f'Global step: {trainer.global_step}\n'
      '----------------------------------------------------------------------------------------------------------------'

    )

    self._send_to_discord(message)
# classes
