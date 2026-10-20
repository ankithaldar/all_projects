#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the PyTorch Lightning Discord callback.

The callback is a notifier: it must never take a training run down.  These
tests therefore check the three things that matter — that it builds the
right messages, that it does not post when no webhook is configured, and
that a delivery failure is swallowed.
'''

import json
import urllib.error

import pytest

from birdcall.callbacks.pl_callback_discord import DISCORD_URL_ENV
from birdcall.callbacks.pl_callback_discord import DiscordCallback


class _FakeResponse:
  '''A stand-in for the object urlopen returns.'''

  def __enter__(self):
    '''Enters the context manager.

    Returns:
      The response itself.
    '''
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> bool:
    '''Leaves the context manager.

    Args:
      exc_type: The exception type, if any.
      exc_value: The exception value, if any.
      traceback: The traceback, if any.

    Returns:
      False, so an exception is not suppressed.
    '''
    return False

  def read(self) -> bytes:
    '''Reads the body of the response.

    Returns:
      An empty body.
    '''
    return b''


@pytest.fixture(name='callback')
def fixture_callback() -> DiscordCallback:
  '''Builds a callback with no webhook configured.

  Returns:
    The callback.
  '''
  return DiscordCallback()


def test_no_message_is_posted_without_a_url(callback: DiscordCallback,
                                            monkeypatch) -> None:
  '''A missing webhook is not an error, it just means no message.

  Args:
    callback: The callback under test.
    monkeypatch: The monkeypatch fixture.
  '''
  monkeypatch.delenv(DISCORD_URL_ENV, raising=False)
  assert callback.post('hello') is False
  assert callback.messages == ['hello']
  assert callback.posted == []


def test_the_message_is_built_per_epoch(callback: DiscordCallback) -> None:
  '''One message is built per training epoch.

  Args:
    callback: The callback under test.
  '''
  trainer = _FakeTrainer({'train_loss': 0.5}, current_epoch=3)
  callback.on_train_epoch_end(trainer, None)
  assert callback.messages == ['[TRAIN] 3: 0.5']


def test_one_message_is_built_per_validation_metric(
    callback: DiscordCallback) -> None:
  '''One message is built per validation metric key.

  Args:
    callback: The callback under test.
  '''
  trainer = _FakeTrainer({'valid_f1score_clip': 0.6, 'valid_lraps': 0.7,
                           'train_loss': 0.5}, current_epoch=2)
  callback.on_validation_end(trainer, None)
  assert callback.messages == ['[VALID-valid_f1score_clip] 2 : 0.6',
                               '[VALID-valid_lraps] 2 : 0.7']


def test_the_message_is_posted_to_the_configured_url(
    callback: DiscordCallback, monkeypatch) -> None:
  '''A configured webhook receives the message.

  Args:
    callback: The callback under test.
    monkeypatch: The monkeypatch fixture.
  '''
  sent = {}

  def fake_urlopen(request, timeout=None):
    '''Records the request instead of sending it.

    Args:
      request: The request that would have been sent.
      timeout: The timeout that would have been used.

    Returns:
      A fake response.
    '''
    sent['url'] = request.full_url
    sent['body'] = json.loads(request.data.decode('utf-8'))
    return _FakeResponse()

  monkeypatch.setenv(DISCORD_URL_ENV, 'https://discord.test/hook')
  monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
  assert callback.post('hello') is True
  assert sent['url'] == 'https://discord.test/hook'
  assert sent['body'] == {'content': 'hello'}
  assert callback.posted == ['hello']


def test_a_delivery_failure_is_not_fatal(callback: DiscordCallback,
                                         monkeypatch) -> None:
  '''A network failure is logged, not raised.

  Args:
    callback: The callback under test.
    monkeypatch: The monkeypatch fixture.
  '''

  def fake_urlopen(request, timeout=None):
    '''Raises the way a broken network would.

    Args:
      request: The request that would have been sent.
      timeout: The timeout that would have been used.

    Raises:
      urllib.error.URLError: Always.
    '''
    raise urllib.error.URLError('down')

  monkeypatch.setenv(DISCORD_URL_ENV, 'https://discord.test/hook')
  monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
  assert callback.post('hello') is False
  assert callback.posted == []


def test_an_explicit_url_wins_over_the_environment(
    callback: DiscordCallback, monkeypatch) -> None:
  '''An explicit URL is used instead of the environment.

  Args:
    callback: The callback under test.
    monkeypatch: The monkeypatch fixture.
  '''
  monkeypatch.setenv(DISCORD_URL_ENV, 'https://from.env/hook')
  callback.url = 'https://explicit.test/hook'
  sent = {}

  def fake_urlopen(request, timeout=None):
    '''Records the request instead of sending it.

    Args:
      request: The request that would have been sent.
      timeout: The timeout that would have been used.

    Returns:
      A fake response.
    '''
    sent['url'] = request.full_url
    return _FakeResponse()

  monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
  assert callback.post('hello') is True
  assert sent['url'] == 'https://explicit.test/hook'


class _FakeTrainer:
  '''The parts of a Lightning trainer the callback touches.'''

  def __init__(self, callback_metrics: dict, current_epoch: int) -> None:
    '''Builds the fake trainer.

    Args:
      callback_metrics: The metrics of the run.
      current_epoch: The current epoch.
    '''
    self.callback_metrics = callback_metrics
    self.current_epoch = current_epoch
