#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Discord webhook notifications for training milestones.

A thin :class:`DiscordNotifier` handles transport (POST to a webhook URL,
never blocking or crashing the run), and :class:`DiscordCallback` maps
Lightning lifecycle events onto concise experiment updates:

* fit start  -> fold/experiment banner
* epoch end  -> macro-AUC progress line
* train end  -> completion + duration + best metric
* exception  -> truncated traceback

The webhook URL is resolved by ``knee.helpers.secrets.get_secret`` from the
name configured in YAML (``DISCORD_WEBHOOK_URL`` by default).
"""

from __future__ import annotations

import time

import pytorch_lightning as pl
import requests
from pytorch_lightning.callbacks import Callback

from knee.helpers.secrets import get_secret
from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

REQUEST_TIMEOUT_S = 10.0
MIN_POST_INTERVAL_S = 2.5
TRACEBACK_SNIPPET_CHARS = 1500


class DiscordNotifier:
  """Rate-limited, failure-tolerant Discord webhook client."""

  def __init__(self, webhook_url: str | None, enabled: bool = True) -> None:
    """Store transport settings.

    Args:
        webhook_url: Full webhook URL; None disables posting.
        enabled: Master switch from configuration.
    """
    self.webhook_url = webhook_url
    self.enabled = enabled and bool(webhook_url)
    self._last_post = 0.0

  def notify(self, message: str) -> bool:
    """Post one markdown message, never raising into training code.

    Args:
        message: Message body (Discord markdown supported).

    Returns:
        True when the POST succeeded.
    """
    if not self.enabled:
      return False
    elapsed = time.time() - self._last_post
    if elapsed < MIN_POST_INTERVAL_S:
      time.sleep(MIN_POST_INTERVAL_S - elapsed)
    try:
      response = requests.post(
        self.webhook_url,
        json={'content': message},
        timeout=REQUEST_TIMEOUT_S,
      )
      self._last_post = time.time()
      ok = response.status_code in (200, 204)
      if not ok:
        _LOGGER.warning(
          'Discord POST %s: %s', response.status_code, response.text[:200]
        )
      return ok
    except requests.RequestException as exc:
      _LOGGER.warning('Discord notification failed: %s', exc)
      return False


def notifier_from_config(config: dict) -> DiscordNotifier:
  """Build a notifier from the composed experiment configuration.

  Args:
      config: Configuration containing ``integrations.discord`` with keys
          ``enabled`` and ``webhook_secret``.

  Returns:
      Configured DiscordNotifier (disabled when secret resolution fails).
  """
  discord_cfg = config.get('integrations', {}).get('discord', {})
  url = get_secret(discord_cfg.get('webhook_secret', 'DISCORD_WEBHOOK_URL'))
  return DiscordNotifier(
    webhook_url=url, enabled=bool(discord_cfg.get('enabled'))
  )


class DiscordCallback(Callback):
  """Forward Lightning lifecycle events to a DiscordNotifier."""

  def __init__(
    self,
    notifier: DiscordNotifier,
    experiment_name: str,
    fold_id: int,
  ) -> None:
    """Bind the callback to one fold's context.

    Args:
        notifier: Shared transport instance.
        experiment_name: Human-readable experiment label.
        fold_id: Fold handled by the active trainer.
    """
    super().__init__()
    self.notifier = notifier
    self.experiment_name = experiment_name
    self.fold_id = fold_id
    self._fit_start = 0.0

  def on_fit_start(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Announce training start.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    del trainer, pl_module
    self._fit_start = time.time()
    self.notifier.notify(
      f'**[{self.experiment_name}]** fold {self.fold_id}: training started'
    )

  def on_validation_epoch_end(
    self,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
  ) -> None:
    """Report the epoch's macro-AUC when available.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    del pl_module
    macro = trainer.callback_metrics.get('val/auc_macro')
    if macro is None:
      return
    self.notifier.notify(
      f'**[{self.experiment_name}]** fold {self.fold_id} '
      f'epoch {trainer.current_epoch}: macro-AUC `{macro:.4f}`'
    )

  def on_train_end(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Announce completion with wall-clock duration.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    del pl_module
    minutes = (time.time() - self._fit_start) / 60.0
    best = trainer.callback_metrics.get('val/auc_macro')
    suffix = f', last macro-AUC `{best:.4f}`' if best is not None else ''
    self.notifier.notify(
      f'**[{self.experiment_name}]** fold {self.fold_id}: finished '
      f'in {minutes:.1f} min{suffix}'
    )

  def on_exception(
    self,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    exception: BaseException,
  ) -> None:
    """Escalate crashes with a traceback snippet.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
        exception: Raised exception instance.
    """
    del trainer, pl_module
    self.notifier.notify(
      f'**[{self.experiment_name}]** fold {self.fold_id}: CRASH '
      f'`{type(exception).__name__}`:\n'
      f'{str(exception)[:TRACEBACK_SNIPPET_CHARS]}'
    )
