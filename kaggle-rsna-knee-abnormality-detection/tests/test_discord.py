#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the Discord notification callback hooks (transport faked)."""

# Simple fake trainers/modules replace full Lightning fits here; hook
# signatures are exercised directly and fixtures shadow module names.
# pylint: disable=protected-access,redefined-outer-name

import http.server
import json
import threading

import pytest

from knee.loggers.discord_logger import (
  DiscordCallback,
  DiscordNotifier,
  notifier_from_config,
)


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


class UnsubscriptableOptimizer:
  """Mimics a bare non-list return: indexing it must fail loudly."""

  def __init__(self, lr: float = 3e-4) -> None:
    self.param_groups = [{'lr': lr}]

  def __getitem__(self, index):  # pragma: no cover - guard assertion aid
    raise TypeError(f'{type(self).__name__} object is not subscriptable')


class BareModule:
  """optimizers() returning ONE unwrapped optimizer (PL single-device)."""

  def __init__(self, lr: float = 2e-4) -> None:
    self._optimizer = UnsubscriptableOptimizer(lr)

  def optimizers(self):
    return self._optimizer


class WrapperOnlyOptimizer:
  """No param_groups; mirrors LightningOptimizer's wrapper surface."""

  def __init__(self, inner) -> None:
    self.optimizer = inner


class WrappedModule:
  """optimizers() returning one Lightning-wrapped optimizer."""

  def __init__(self, lr: float = 1.5e-4) -> None:
    self._inner = UnsubscriptableOptimizer(lr)

  def optimizers(self):
    return WrapperOnlyOptimizer(self._inner)


class BrokenModule:
  """optimizers() raising — LR probe must swallow and degrade."""

  def optimizers(self):
    raise RuntimeError('not configured')


@pytest.fixture()
def callback() -> tuple[DiscordCallback, FakeNotifier]:
  notifier = FakeNotifier()
  cb = DiscordCallback(
    notifier=notifier,
    experiment_name='exp',
    fold_id=2,
    step_interval=50,
    first_step_ping=False,  # interval semantics under test
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


def test_lr_probe_survives_all_lightning_optimizer_shapes(callback):
  """Regression: bare/wrapped/raising optimizers must not crash training.

  Lightning returns a single LightningOptimizer (unsubscriptable!) in
  single-device runs; the probe previously died on ``items[0]`` and
  took training down with it at every heartbeat step.
  """
  cb, notifier = callback
  for module in (BareModule(), WrappedModule(), BrokenModule()):
    trainer = FakeTrainer(metrics={'train/loss': 0.5})
    trainer.global_step = 50  # heartbeat fires on non-zero multiples
    cb.on_train_batch_end(trainer, module, None, None, 0)
  assert all('lr `' in m or 'train_loss' in m for m in notifier.messages)
  # Bare and wrapped shapes still surface a numeric lr; broken degrades.
  lr_lines = [m for m in notifier.messages if 'lr `' in m]
  assert len(lr_lines) >= 2
  assert any('1.50e-04' in line for line in lr_lines)  # wrapped inner
  assert any('2.00e-04' in line for line in lr_lines)  # bare optimizer


def test_lr_probe_tolerates_empty_lists(callback):
  class EmptyModule:
    def optimizers(self):
      return []

  cb, notifier = callback
  trainer = FakeTrainer(metrics={'train/loss': 0.25})
  cb.on_train_batch_end(trainer, EmptyModule(), None, None, 49)
  assert notifier.messages == []  # not an interval multiple yet
  trainer.global_step = 50
  cb.on_train_batch_end(trainer, EmptyModule(), None, None, 0)
  assert 'lr' not in notifier.messages[0]
  assert 'train_loss `0.2500`' in notifier.messages[0]


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


class TestRealDispatch:
  """Loopback HTTP proves transport end-to-end through requests."""

  def test_posts_reach_webhook(self, monkeypatch):
    delivered: list[str] = []

    class Sink(http.server.BaseHTTPRequestHandler):
      # pylint: disable=invalid-name  (stdlib HTTP verb contract)
      def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        delivered.append(json.loads(body)['content'])
        self.send_response(204)
        self.end_headers()

      def log_message(self, *args):
        del args

    try:
      server = http.server.HTTPServer(('127.0.0.1', 0), Sink)
    except OSError:
      pytest.skip('loopback unavailable')
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv('DISCORD_WEBHOOK_URL', f'http://127.0.0.1:{port}/x')
    try:
      config = {
        'integrations': {
          'discord': {'enabled': True, 'webhook_secret': 'DISCORD_WEBHOOK_URL'}
        }
      }
      live = notifier_from_config(config)
      assert live.enabled
      assert live.notify('**[exp]** cache: started building HDF5')
      assert live.notify('step line')
      server.shutdown()
    finally:
      pass
    assert any('started building HDF5' in m for m in delivered)
    assert '**[exp]** cache:' in delivered[0]

  def test_unresolved_secret_is_loud_and_disabled(self, caplog, monkeypatch):
    from knee.loggers import (  # pylint: disable=import-outside-toplevel
      discord_logger as dl,
    )

    monkeypatch.delenv('DISCORD_WEBHOOK_URL', raising=False)
    monkeypatch.setattr(dl, 'get_secret', lambda *a, **k: None)
    with caplog.at_level('WARNING'):
      notifier = dl.notifier_from_config(
        {'integrations': {'discord': {'enabled': True}}}
      )
    assert not notifier.enabled
    assert 'will NOT be sent' in caplog.text
    assert 'DISCORD_WEBHOOK_URL' in caplog.text


def test_first_step_ping_gives_instant_pace():
  """Pace signal arrives at step 1, not a full 50-step interval later."""
  notifier = FakeNotifier()
  cb = DiscordCallback(
    notifier=notifier,
    experiment_name='exp',
    fold_id=2,
    step_interval=50,
    first_step_ping=True,
  )
  trainer = FakeTrainer(metrics={'train/loss': 0.9})
  trainer.global_step = 1
  module = FakeModule()
  cb.on_train_batch_end(trainer, module, None, None, 0)
  assert len(notifier.messages) == 1
  assert 'first optimizer step - pacing OK' in notifier.messages[0]
  assert 'step 1' in notifier.messages[0]
  # Step 2 is not a multiple and not step 1: silence.
  trainer.global_step = 2
  cb.on_train_batch_end(trainer, module, None, None, 1)
  assert len(notifier.messages) == 1
