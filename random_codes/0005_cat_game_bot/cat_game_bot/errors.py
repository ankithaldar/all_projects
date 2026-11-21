#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Exception types shared across the cat_game_bot package."""

__all__ = ('AdbError', 'ConfigError', 'VisionError', 'DbError')


class AdbError(RuntimeError):
  """Raised when an ADB operation fails or times out."""


class ConfigError(ValueError):
  """Raised when command-line configuration is invalid or unsafe."""


class VisionError(RuntimeError):
  """Raised when screen capture or image decoding fails."""


class DbError(RuntimeError):
  """Raised when the interaction log database cannot be used."""
