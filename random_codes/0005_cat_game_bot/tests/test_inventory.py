#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for OCR-based Bag inventory counting."""

import numpy as np

import cat_game_bot.config as config
from cat_game_bot.inventory import CANONICAL_IDS, InventoryReader
from cat_game_bot.models import Match, Rect, Screen
from cat_game_bot.targets import item_template


def test_canonical_ids_are_the_single_config_source():
  """inventory re-exports the single canonical id tuple from config."""
  assert CANONICAL_IDS is config.CANONICAL_IDS


def _reader(observed, fake_ocr, fake_matcher, fake_library,
            names=None, extra_matches=None):
  """Build an InventoryReader from id -> OCR text (None hides the icon)."""
  matches = {}
  texts = {}
  for idx, (item_id, text) in enumerate(observed.items()):
    if text is None:
      continue
    icon = Rect((idx % 4) * 250, (idx // 4) * 150, 50, 50)
    name = item_template(item_id)
    matches[name] = [Match(name, icon, 0.9)]
    count = Rect(icon.x, icon.y + 50, 40, 25)
    texts[(count.x, count.y, count.width, count.height)] = text
  if extra_matches:
    matches.update(extra_matches)
  if names is None:
    names = [item_template(item_id) for item_id in CANONICAL_IDS]
  offsets = {f'item_count.{item_id}': (0, 50, 40, 25)
             for item_id in CANONICAL_IDS}
  return InventoryReader(fake_ocr(texts), fake_matcher(matches),
                         offsets, fake_library(names))


def _screen(screen_h=2400):
  """Return a minimal blank Bag screen."""
  return Screen(np.zeros((screen_h, 1080, 3), dtype=np.uint8))


def test_read_returns_the_86_69_59_18_fixture(fake_ocr, fake_matcher,
                                              fake_library):
  """The documented Bag counts are read from their icon-adjacent cells."""
  reader = _reader({'string': '86', 'wood': '69', 'ribbon': '59',
                    'metal': '18'}, fake_ocr, fake_matcher, fake_library)
  counts = reader.read(_screen())
  assert counts['string'] == 86
  assert counts['wood'] == 69
  assert counts['ribbon'] == 59
  assert counts['metal'] == 18


def test_read_includes_all_23_canonical_ids(fake_ocr, fake_matcher,
                                            fake_library):
  """Every canonical id is present even when its icon is absent."""
  counts = _reader({}, fake_ocr, fake_matcher, fake_library).read(_screen())
  assert set(counts) == set(CANONICAL_IDS)
  assert len(counts) == 23


def test_missing_icon_yields_zero(fake_ocr, fake_matcher, fake_library):
  """Hidden zero items have no icon match and count as zero."""
  reader = _reader({'needles': None, 'string': '86'}, fake_ocr,
                   fake_matcher, fake_library)
  counts = reader.read(_screen())
  assert counts['needles'] == 0
  assert counts['string'] == 86


def test_unreadable_count_yields_zero(fake_ocr, fake_matcher, fake_library):
  """Garbage OCR text maps to a zero count."""
  counts = _reader({'string': '???'}, fake_ocr, fake_matcher,
                   fake_library).read(_screen())
  assert counts['string'] == 0


def test_count_suffix_is_parsed(fake_ocr, fake_matcher, fake_library):
  """K-suffixed counts are expanded."""
  counts = _reader({'string': '12.1K'}, fake_ocr, fake_matcher,
                   fake_library).read(_screen())
  assert counts['string'] == 12100


def test_out_of_bounds_count_rect_yields_zero(fake_ocr, fake_matcher,
                                              fake_library):
  """A count cell that does not fit on screen is ignored."""
  icon = Rect(0, 2380, 50, 50)
  match = Match(item_template('string'), icon, 0.9)
  reader = _reader({'string': '86'}, fake_ocr, fake_matcher, fake_library,
                   extra_matches={item_template('string'): [match]})
  counts = reader.read(_screen(screen_h=2400))
  assert counts['string'] == 0


def test_highest_score_match_wins(fake_ocr, fake_matcher, fake_library):
  """When an icon matches twice, the strongest match supplies the count."""
  name = item_template('string')
  low = Match(name, Rect(0, 0, 50, 50), 0.6)
  high = Match(name, Rect(500, 0, 50, 50), 0.95)
  ocr = fake_ocr({(500, 50, 40, 25): '99', (0, 50, 40, 25): '5'})
  matcher = fake_matcher({name: [low, high]})
  offsets = {'item_count.string': (0, 50, 40, 25)}
  reader = InventoryReader(ocr, matcher, offsets, fake_library([name]))
  assert reader.read(_screen())['string'] == 99


def test_noncanonical_icon_name_is_ignored(fake_ocr, fake_matcher,
                                           fake_library):
  """Only the canonical ``item_<id>`` name is resolved; fallbacks are gone."""
  name = 'item_icon_string'
  match = Match(name, Rect(0, 0, 50, 50), 0.9)
  reader = InventoryReader(fake_ocr({(0, 50, 40, 25): '86'}),
                           fake_matcher({name: [match]}),
                           {'item_count.string': (0, 50, 40, 25)},
                           fake_library([name]))
  assert reader.read(_screen())['string'] == 0


def test_all_canonical_icon_names_exist_in_manifest(fake_ocr, fake_matcher,
                                                     fake_library,
                                                     manifest_names):
  """Each canonical id resolves to a real icon in the committed manifest."""
  reader = _reader({'silver': '12'}, fake_ocr, fake_matcher, fake_library,
                   names=list(manifest_names))
  canonical = [item_template(item_id) for item_id in CANONICAL_IDS]
  assert frozenset(canonical) <= manifest_names
  assert reader.read(_screen())['silver'] == 12
