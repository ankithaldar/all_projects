#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the cat_game_bot.errors exception hierarchy."""

import cat_game_bot.errors as errors


def test_adb_error_is_runtime_error():
  """AdbError derives from RuntimeError."""
  assert issubclass(errors.AdbError, RuntimeError)


def test_config_error_is_value_error():
  """ConfigError derives from ValueError."""
  assert issubclass(errors.ConfigError, ValueError)


def test_vision_error_is_runtime_error():
  """VisionError derives from RuntimeError."""
  assert issubclass(errors.VisionError, RuntimeError)


def test_db_error_is_runtime_error():
  """DbError derives from RuntimeError."""
  assert issubclass(errors.DbError, RuntimeError)


def test_errors_carry_message():
  """Each exception preserves the message passed to its constructor."""
  for error_type in (errors.AdbError, errors.ConfigError,
                     errors.VisionError, errors.DbError):
    exc = error_type('boom')
    assert str(exc) == 'boom'
