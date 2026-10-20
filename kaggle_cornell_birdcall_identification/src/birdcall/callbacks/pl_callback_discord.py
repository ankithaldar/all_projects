#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The Discord notification callback.

The original's print logger posted to a Slack webhook: one message per
training epoch, carrying the epoch's loss, and **one message per metric
key** at the end of every validation.  The same cadence is kept here, on
Discord.

The webhook URL is read at call time rather than at import time, which
fixes the original's wart: the environment could not be changed after the
import.  A missing URL is not an error, it only means no message is sent.
'''

import json
import logging
import os
import urllib.error
import urllib.request

from lightning.pytorch import Callback

LOGGER = logging.getLogger(__name__)

DISCORD_URL_ENV = 'DISCORD_WEBHOOK_URL'
VALID_PREFIX = 'valid_'
TIMEOUT = 10
MAX_CONTENT = 2000


class DiscordCallback(Callback):
  '''Posts the epoch and the validation metrics to a Discord webhook.

  Attributes:
    url: The webhook URL, or ``None`` to read it from the environment
      at every call.
    messages: Every message the callback built.
    posted: The messages that were actually delivered.
  '''

  def __init__(self, url: str | None = None) -> None:
    '''Builds the callback.

    Args:
      url: The webhook URL.  When it is ``None`` the environment is
        read at every call.
    '''
    super().__init__()
    self.url = url
    self.messages: list[str] = []
    self.posted: list[str] = []

  def on_train_epoch_end(self, trainer, pl_module) -> None:
    '''Posts one message per training epoch.

    Args:
      trainer: The trainer.
      pl_module: The module of the run.
    '''
    loss = trainer.callback_metrics.get('train_loss')
    self.post(f'[TRAIN] {trainer.current_epoch}: {loss}')

  def on_validation_end(self, trainer, pl_module) -> None:
    '''Posts one message per validation metric key.

    Args:
      trainer: The trainer.
      pl_module: The module of the run.
    '''
    for key, value in sorted(trainer.callback_metrics.items()):
      if not key.startswith(VALID_PREFIX):
        continue
      self.post(f'[VALID-{key}] {trainer.current_epoch} : {value}')

  def post(self, message: str) -> bool:
    '''Posts one message.

    Args:
      message: The text of the message.

    Returns:
      Whether the message was delivered.
    '''
    self.messages.append(message)
    url = self.url or os.environ.get(DISCORD_URL_ENV)
    if not url:
      LOGGER.debug('No Discord webhook URL is configured, so no message is '
                   'sent.')
      return False
    payload = json.dumps({'content': message[:MAX_CONTENT]}).encode('utf-8')
    request = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'})
    try:
      with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        response.read()
    except (urllib.error.URLError, OSError, ValueError) as error:
      LOGGER.warning('Could not post to Discord: %s', error)
      return False
    self.posted.append(message)
    return True
