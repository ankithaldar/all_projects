#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Discord webhook notifications for training milestones.

A thin :class:`DiscordNotifier` handles transport (POST to a webhook URL,
never blocking or crashing the run), and :class:`DiscordCallback` maps
Lightning lifecycle events onto concise experiment updates:

* fit start  -> fold/experiment banner
* every N optimizer steps -> live train-loss / lr heartbeat
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

  Resolution failures are deliberately LOUD: training has been burned
  by silent no-op notifiers (symptom: zero Discord messages while runs
  progress normally). Reasons are logged at WARNING so kernel logs and
  notebook scrollback surface them even when nothing ever posts.

  Args:
      config: Configuration containing ``integrations.discord`` with keys
          ``enabled`` and ``webhook_secret``.

  Returns:
      Configured DiscordNotifier (disabled when secret resolution fails).
  """
  discord_cfg = config.get('integrations', {}).get('discord', {})
  secret_name = discord_cfg.get('webhook_secret', 'DISCORD_WEBHOOK_URL')
  url = get_secret(secret_name)
  if not discord_cfg.get('enabled'):
    _LOGGER.info('Discord integration OFF: integrations.discord.enabled=false')
    return DiscordNotifier(webhook_url=None, enabled=False)
  if not url:
    _LOGGER.warning(
      'Discord notifications will NOT be sent: webhook secret %r '
      'resolved empty across os.environ / .env / Kaggle User Secrets',
      secret_name,
    )
    return DiscordNotifier(webhook_url=None, enabled=True)
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
    step_interval: int = 50,
  ) -> None:
    """Bind the callback to one fold's context.

    Args:
        notifier: Shared transport instance.
        experiment_name: Human-readable experiment label.
        fold_id: Fold handled by the active trainer.
        step_interval: Post a metrics heartbeat every this many optimizer
            steps (``trainer.global_step``, accumulation-aware).
    """
    super().__init__()
    self.notifier = notifier
    self.experiment_name = experiment_name
    self.fold_id = fold_id
    self.step_interval = max(1, int(step_interval))
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

  @staticmethod
  def _current_lr(pl_module: pl.LightningModule) -> float | None:
    """Read the live learning rate from the first optimizer group.

    Tolerates every shape ``pl_module.optimizers()`` takes across
    Lightning versions: a bare optimizer, a ``LightningOptimizer``
    wrapper, or lists of either. Any surprise here must degrade to
    'no lr shown', never break training.

    Args:
        pl_module: Active module.

    Returns:
        Learning rate, or None when no optimizer/param-group exists yet.
    """
    try:
      returned = pl_module.optimizers()
      items = returned if isinstance(returned, (list, tuple)) else [returned]
      items = [item for item in items if item is not None]
      if not items:
        return None
      # LightningOptimizer wrappers carry the raw module under .optimizer.
      raw = getattr(items[0], 'optimizer', items[0])
      groups = getattr(raw, 'param_groups', [])
      lr = groups[0].get('lr') if groups else None
      return float(lr) if lr is not None else None
    except Exception:  # pylint: disable=broad-except
      # Metrics are auxiliary: ANY optimizer-shape surprise must degrade
      # to 'no lr shown' instead of killing the fit.
      return None

  def on_train_batch_end(
    self,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    outputs: dict | None,
    batch: object,
    batch_idx: int,
  ) -> None:
    """Post a heartbeat with train loss/lr every ``step_interval`` steps.

    Uses ``trainer.global_step`` so gradient accumulation is honored:
    the counter only advances on real optimizer updates.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
        outputs: Step output (unused; loss comes from callback_metrics).
        batch: Current batch (unused).
        batch_idx: Index of the batch within the epoch (unused).
    """
    del outputs, batch, batch_idx
    step = int(trainer.global_step)
    if step == 0 or step % self.step_interval != 0:
      return
    loss = trainer.callback_metrics.get('train/loss')
    parts = [f'step {step} (epoch {trainer.current_epoch})']
    if loss is not None:
      parts.append(f'train_loss `{float(loss):.4f}`')
    lr = self._current_lr(pl_module)
    if lr is not None:
      parts.append(f'lr `{lr:.2e}`')
    self.notifier.notify(
      f'**[{self.experiment_name}]** fold {self.fold_id}: ' + ', '.join(parts)
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
    if trainer.sanity_checking:
      return
    macro = trainer.callback_metrics.get('val/auc_macro')
    if macro is None:
      return
    self.notifier.notify(
      f'**[{self.experiment_name}]** fold {self.fold_id} '
      f'epoch {trainer.current_epoch}: macro-AUC `{macro:.4f}`'
    )

  def on_train_epoch_end(
    self,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
  ) -> None:
    """Fallback epoch-end ping when validation did not report this epoch.

    Fires only when no macro-AUC message was just emitted by
    :meth:`on_validation_epoch_end`, keeping exactly one Discord line per
    completed epoch.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    del pl_module
    if trainer.sanity_checking:
      return
    if 'val/auc_macro' in trainer.callback_metrics:
      return
    loss = trainer.callback_metrics.get('train/loss')
    detail = f', train_loss `{float(loss):.4f}`' if loss is not None else ''
    self.notifier.notify(
      f'**[{self.experiment_name}]** fold {self.fold_id} '
      f'epoch {trainer.current_epoch} complete{detail}'
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
