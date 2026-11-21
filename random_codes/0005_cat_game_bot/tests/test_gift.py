#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for timer-ready gift collection and coin verification."""

import numpy as np
import pytest

from cat_game_bot.gift import GiftCollector
from cat_game_bot.models import (
    Action,
    ActionKind,
    Rect,
    Screen,
)

COINS = Rect(0, 0, 200, 50)
TIMER = Rect(100, 100, 50, 50)


def _screen():
  """Return a minimal synthetic screen."""
  return Screen(np.zeros((2400, 1080, 3), dtype=np.uint8))


def _key(rect):
  return (rect.x, rect.y, rect.width, rect.height)


def _collector(texts):
  """Build a GiftCollector with an OCR stub keyed by rect."""
  class FakeOcr:
    def read(self, unused_screen, rect):
      return texts.get(_key(rect), '')

  return GiftCollector(FakeOcr(), {'coins': COINS, 'gift_timer': TIMER})


@pytest.mark.parametrize('text', [
    'READY', 'GET COINS IN THE BANK!', 'garbage', '???', '#@$%',
])
def test_is_ready_false_for_unparseable_timer_text(text):
  """A covered timer yields unparseable text and must not open a gift cycle."""
  assert not _collector({_key(TIMER): text}).is_ready(_screen())


def test_is_ready_false_for_blank_timer_with_unreadable_coins():
  """A splash-like screen (blank timer, no coins) must not open a gift cycle."""
  data = {_key(TIMER): '', _key(COINS): ''}
  assert not _collector(data).is_ready(_screen())


def test_is_ready_true_for_blank_timer_with_readable_coins():
  """No countdown text with readable coins means the gift is ready."""
  data = {_key(TIMER): '', _key(COINS): '1,234'}
  assert _collector(data).is_ready(_screen())


def test_is_ready_true_for_zero_clock_timer_with_readable_coins():
  """An elapsed 00:00 countdown with readable coins means the gift is ready."""
  data = {_key(TIMER): '00:00', _key(COINS): '1,234'}
  assert _collector(data).is_ready(_screen())


def test_is_ready_false_for_seconds_countdown():
  """A seconds countdown suppresses gift readiness."""
  assert not _collector({_key(TIMER): '35s'}).is_ready(_screen())


def test_is_ready_false_for_clock_countdown():
  """A mm:ss countdown suppresses gift readiness."""
  assert not _collector({_key(TIMER): '04:57'}).is_ready(_screen())


def test_plan_returns_tap_at_timer_center_when_ready():
  """plan produces the single icon-tap action when a gift is ready."""
  data = {_key(TIMER): '', _key(COINS): '1,234'}
  action = _collector(data).plan(_screen())
  assert isinstance(action, Action)
  assert action.kind is ActionKind.TAP
  assert action.point == TIMER.center()
  assert action.end is None
  assert action.reason == 'gift_icon'


def test_plan_returns_none_when_not_ready():
  """plan produces no action while a countdown is active."""
  assert _collector({_key(TIMER): '35s'}).plan(_screen()) is None


def test_plan_reads_timer_itself_and_has_no_ready_parameter():
  """plan re-reads the timer and no longer accepts a ready override."""
  calls = []

  class CountingOcr:
    """OCR stub recording which regions were read."""

    def read(self, unused_screen, rect):
      calls.append(_key(rect))
      return '1234' if _key(rect) == _key(COINS) else ''

  collector = GiftCollector(
    CountingOcr(), {'coins': COINS, 'gift_timer': TIMER})
  action = collector.plan(_screen())
  assert action is not None
  assert action.reason == 'gift_icon'
  assert calls.count(_key(TIMER)) == 1
  with pytest.raises(TypeError):
    collector.plan(_screen(), ready=True)


@pytest.mark.parametrize('text, expected', [
    ('1,234', 1234), ('12.1K', 12100), ('9K', 9000), ('1.5M', 1500000),
    ('garbage', None), ('', None),
])
def test_read_coins_parses_counts(text, expected):
  """read_coins delegates to parse_count for comma and K/M suffixes."""
  assert _collector({_key(COINS): text}).read_coins(_screen()) == expected


def test_verify_ok_for_valid_delta():
  """A bounded positive delta is accepted."""
  result = _collector({}).verify(6423826, 6434986)
  assert result.ok
  assert result.delta == 11160


def test_verify_fails_when_before_is_none():
  """A missing before reading fails verification."""
  result = _collector({}).verify(None, 100)
  assert not result.ok
  assert result.delta == 0


def test_verify_fails_when_after_is_none():
  """A missing after reading fails verification."""
  result = _collector({}).verify(100, None)
  assert not result.ok
  assert result.delta == 0


@pytest.mark.parametrize('before, after', [(100, 100), (100, 50)])
def test_verify_fails_for_non_positive_delta(before, after):
  """Zero and negative deltas are rejected."""
  assert not _collector({}).verify(before, after).ok


def test_verify_fails_for_implausible_delta():
  """A delta above 100000 is rejected."""
  result = _collector({}).verify(100, 100101)
  assert not result.ok
  assert result.delta == 100001
