#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the Discord notification callback hooks (transport faked)."""

# Simple fake trainers/modules replace full Lightning fits here; hook
# signatures are exercised directly and fixtures shadow module names.
# pylint: disable=protected-access,redefined-outer-name

import pytest

from knee.loggers.discord_logger import DiscordCallback, DiscordNotifier


class FakeNotifier(DiscordNotifier):
  """Capture messages instead of POSTing anywhere."""

  def __init__(self) -> None:
    super().__init__(webhook_url=None, enabled=True)
    self.messages: list[str] = []

  def notify(self, message: str) -> bool:
    self.messages.append(message)
    return True


class FakeTrainer:
  """Minimal stand-in exposing only what the callbacks read."""

  def __init__(
    self,
    global_step: int = 0,
    current_epoch: int = 0,
    metrics: dict | None = None,
    sanity_checking: bool = False,
  ) -> None:
    self.global_step = global_step
    self.current_epoch = current_epoch
    self.callback_metrics = metrics or {}
    self.sanity_checking = sanity_checking


class FakeOptimizer:
  """Param-group carrier mimicking torch.optim semantics."""

  def __init__(self, lr: float = 3e-4) -> None:
    self.param_groups = [{'lr': lr}]


class FakeModule:
  """Module surface used by the LR probe."""

  def __init__(self, lr: float = 3e-4) -> None:
    self._optimizer = FakeOptimizer(lr)

  def optimizers(self):
    return [self._optimizer]


@pytest.fixture()
def callback() -> tuple[DiscordCallback, FakeNotifier]:
  notifier = FakeNotifier()
  cb = DiscordCallback(
    notifier=notifier,
    experiment_name='exp',
    fold_id=2,
    step_interval=50,
  )
  return cb, notifier


def _post_batch(cb: DiscordCallback, trainer: FakeTrainer, step: int) -> None:
  """Drive on_train_batch_end exactly as Lightning would."""
  trainer.global_step = step
  metrics = getattr(trainer, 'callback_metrics')
  if 'train/loss' not in metrics and step > 0:
    metrics['train/loss'] = 0.25 + step / 10000.0
  module = FakeModule(lr=2.5e-4 - step * 1e-9)
  cb.on_train_batch_end(
    trainer, module, outputs=None, batch=None, batch_idx=step % 7
  )


def test_heartbeat_fires_on_interval_multiples(callback):
  cb, notifier = callback
  trainer = FakeTrainer(metrics={'train/loss': 0.31})
  for step in (1, 49, 51, 99):
    _post_batch(cb, trainer, step)
  assert notifier.messages == []

  _post_batch(cb, trainer, 50)
  assert len(notifier.messages) == 1
  assert 'step 50' in notifier.messages[0]
  assert '`0.3100`' in notifier.messages[0]

  _post_batch(cb, trainer, 100)
  assert len(notifier.messages) == 2
  assert 'step 100' in notifier.messages[1]


def test_heartbeat_includes_lr_and_skips_step_zero(callback):
  cb, notifier = callback
  trainer = FakeTrainer(metrics={'train/loss': 0.5})
  _post_batch(cb, trainer, 0)
  assert notifier.messages == []
  _post_batch(cb, trainer, 50)
  assert 'lr `2.50e-04`' in notifier.messages[0]


def test_validation_epoch_reports_macro_auc(callback):
  cb, notifier = callback
  trainer = FakeTrainer(current_epoch=3, metrics={'val/auc_macro': 0.81234})
  cb.on_validation_epoch_end(trainer, FakeModule())
  assert notifier.messages == ['**[exp]** fold 2 epoch 3: macro-AUC `0.8123`']


def test_validation_epoch_silent_without_metric_or_sanity(callback):
  cb, notifier = callback
  cb.on_validation_epoch_end(FakeTrainer(), FakeModule())
  cb.on_validation_epoch_end(FakeTrainer(sanity_checking=True), FakeModule())
  assert notifier.messages == []


def test_epoch_end_falls_back_without_val_metric(callback):
  """No validation run -> still exactly one line per completed epoch."""
  cb, notifier = callback
  trainer = FakeTrainer(current_epoch=1, metrics={'train/loss': 0.42})
  cb.on_train_epoch_end(trainer, FakeModule())
  assert notifier.messages == [
    '**[exp]** fold 2 epoch 1 complete, train_loss `0.4200`'
  ]


def test_no_double_ping_when_validation_reported(callback):
  cb, notifier = callback
  trainer = FakeTrainer(current_epoch=2, metrics={'val/auc_macro': 0.9})
  cb.on_validation_epoch_end(trainer, FakeModule())
  cb.on_train_epoch_end(trainer, FakeModule())
  assert len(notifier.messages) == 1


def test_lifecycle_messages_flow_through_notifier(callback):
  cb, notifier = callback
  cb.on_fit_start(FakeTrainer(), FakeModule())
  cb.on_exception(FakeTrainer(), FakeModule(), ValueError('boom'))
  assert 'training started' in notifier.messages[0]
  assert 'CRASH' in notifier.messages[1]
  assert 'ValueError' in notifier.messages[1]
