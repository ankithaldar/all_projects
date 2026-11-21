#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OCR reader abstraction and parsers for in-game counts and durations."""

import re
from typing import Protocol

import pytesseract

from cat_game_bot.models import Rect, Screen

__all__ = (
  'OcrReader', 'TesseractReader', 'parse_count', 'parse_duration',
)

_COUNT_RE = re.compile(r'^([0-9][0-9,]*(?:\.[0-9]+)?)\s*([KkMm])?$')
_CLOCK_RE = re.compile(r'^(\d+):([0-5]?\d)(?::([0-5]?\d))?$')
_UNITS_RE = re.compile(
  r'^(?:(\d+)\s*[hH])?\s*(?:(\d+)\s*[mM])?\s*(?:(\d+)\s*[sS])?$')


class OcrReader(Protocol):
  """Structural interface for a reader that extracts text from a rect."""

  def read(self, screen: Screen, rect: Rect) -> str:
    """Return the text found inside the given screen rectangle.

    Args:
      screen: The screen frame to read.
      rect: The region of interest.

    Returns:
      The recognised text.
    """
    ...


class TesseractReader:
  """OCR reader backed by the pytesseract command-line wrapper."""

  def __init__(self, tesseract_cmd: str, lang: str = 'eng',
               psm: int = 7) -> None:
    """Initialise the reader with a binary path and OCR settings.

    Args:
      tesseract_cmd: Path to the Tesseract binary.
      lang: Tesseract language code.
      psm: Tesseract page segmentation mode.
    """
    self._tesseract_cmd = tesseract_cmd
    self._lang = lang
    self._psm = psm

  def read(self, screen: Screen, rect: Rect) -> str:
    """Read stripped text from a rectangle of the screen.

    Args:
      screen: The screen frame to read.
      rect: The region of interest.

    Returns:
      The recognised text with surrounding whitespace removed.
    """
    region = screen.pixels[rect.y:rect.y + rect.height,
                           rect.x:rect.x + rect.width]
    pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
    text = pytesseract.image_to_string(
      region, lang=self._lang, config=f'--psm {self._psm}')
    return text.strip()


def parse_count(text: str) -> int | None:
  """Parse an OCR count with optional commas and K/M suffixes.

  Args:
    text: Recognised count text.

  Returns:
    The integer count, or None when the text is unparseable.
  """
  match = _COUNT_RE.fullmatch(text.strip())
  if match is None:
    return None
  try:
    number = float(match.group(1).replace(',', ''))
    suffix = match.group(2)
    if suffix in ('K', 'k'):
      number *= 1000
    elif suffix in ('M', 'm'):
      number *= 1000000
    return int(round(number))
  except (OverflowError, ValueError):
    return None


def parse_duration(text: str) -> float | None:
  """Parse an OCR duration in clock or h/m/s notation into seconds.

  Args:
    text: Recognised duration text.

  Returns:
    The duration in seconds, or None when the text is unparseable or a
    numeric conversion overflows.
  """
  cleaned = text.strip()
  try:
    clock = _CLOCK_RE.fullmatch(cleaned)
    if clock is not None:
      parts = [int(value) for value in clock.groups() if value is not None]
      if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
      return float(parts[0] * 3600 + parts[1] * 60 + parts[2])

    units = _UNITS_RE.fullmatch(cleaned)
    if units is None or all(value is None for value in units.groups()):
      return None
    hours, minutes, seconds = (
      int(value) if value is not None else 0 for value in units.groups())
    return float(hours * 3600 + minutes * 60 + seconds)
  except (OverflowError, ValueError):
    return None
