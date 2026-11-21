#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for cat_game_bot.ocr Tesseract reader and text parsers."""

import numpy as np
import pytest

import cat_game_bot.ocr as ocr
from cat_game_bot.models import Rect, Screen


@pytest.mark.parametrize('text,expected', [
  ('86', 86),
  ('0', 0),
  (' 1,234 ', 1234),
  ('1.5K', 1500),
  ('12k', 12000),
  ('2M', 2000000),
  ('1,234K', 1234000),
])
def test_parse_count_valid(text, expected):
  """parse_count handles plain, comma-separated and K/M suffixed values."""
  assert ocr.parse_count(text) == expected


@pytest.mark.parametrize('text', ['', 'abc', '-1', '1 2', 'K', '1KK', '1.2G'])
def test_parse_count_invalid(text):
  """parse_count returns None for unparseable text."""
  assert ocr.parse_count(text) is None


@pytest.mark.parametrize('text', ['9' * 400, '9' * 400 + 'K'])
def test_parse_count_overflow_returns_none(text):
  """An oversized digit run returns None instead of raising OverflowError."""
  assert ocr.parse_count(text) is None


@pytest.mark.parametrize('text,expected', [
  ('35s', 35.0),
  ('04:57', 297.0),
  ('1h 57m', 7020.0),
  ('1h57m', 7020.0),
  ('1:02:03', 3723.0),
  ('2m 5s', 125.0),
  ('5m', 300.0),
])
def test_parse_duration_valid(text, expected):
  """parse_duration handles seconds, clock and h/m/s forms."""
  assert ocr.parse_duration(text) == pytest.approx(expected)


@pytest.mark.parametrize('text', ['', 'abc', '1h 2', '-5s', '::'])
def test_parse_duration_invalid(text):
  """parse_duration returns None for unparseable text."""
  assert ocr.parse_duration(text) is None


@pytest.mark.parametrize('text', [
  '9' * 400 + ':00',
  '9' * 5000 + ':00',
  '9' * 5000 + 's',
])
def test_parse_duration_oversized_returns_none(text):
  """Oversized clock and seconds fields return None instead of raising."""
  assert ocr.parse_duration(text) is None


def test_tesseract_reader_crops_and_configures(monkeypatch):
  """TesseractReader crops the rect and invokes pytesseract with psm/lang."""
  calls = {}
  pixels = np.zeros((100, 100, 3), dtype=np.uint8)

  def fake_image_to_string(image, lang, config):
    calls['shape'] = image.shape
    calls['lang'] = lang
    calls['config'] = config
    return '  86  \n'

  monkeypatch.setattr(ocr.pytesseract, 'image_to_string',
                      fake_image_to_string)
  reader = ocr.TesseractReader('/opt/bin/tesseract')
  result = reader.read(Screen(pixels), Rect(10, 20, 30, 40))
  assert result == '86'
  assert calls['shape'] == (40, 30, 3)
  assert calls['lang'] == 'eng'
  assert calls['config'] == '--psm 7'
  assert ocr.pytesseract.pytesseract.tesseract_cmd == '/opt/bin/tesseract'
