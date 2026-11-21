#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ordered screen-state detection and loading fallbacks."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from cat_game_bot.detection import (
  BANNER_MARKERS,
  LOADING_TEXT,
  POPUP_MARKERS,
  ScreenDetector,
  is_loading,
)
from cat_game_bot.models import ActionKind, GameState, Match, Rect, Screen
from cat_game_bot.navigator import Memory, Navigator
from cat_game_bot.vision import (
  TEMPLATE_SCORE_OVERRIDES,
  TemplateLibrary,
  TemplateMatcher,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / 'templates' / 'manifest.json'
FIXTURE_PATH = (
    Path(__file__).resolve().parent / 'fixtures' / 'coins_popup.png')
FRESH_FRUIT_FIXTURE = (
    Path(__file__).resolve().parent / 'fixtures' / 'fresh_fruit_popup.png')
FRESH_FRUIT_BANNER = 'banner_fresh_fruit'
F23_TEMPLATES = (
    'banner_coins', 'gift_collect', 'banner_flash_sale', 'banner_vip',
    'banner_club_token', 'banner_harvest', FRESH_FRUIT_BANNER,
    'close_dialog', 'crafting_marker', 'home_marker',
)


def _match(name, score=0.9):
  return Match(name, Rect(10, 20, 30, 40), score)


class FakeMatcher:
  def __init__(self, found=None):
    self.found = found or {}

  def match(self, unused_screen, name):
    return self.found.get(name, [])

  def match_all(self, unused_screen):
    return [match for group in self.found.values() for match in group]


def _screen(color=255):
  arr = np.full((2400, 1080, 3), color, dtype=np.uint8)
  return Screen(arr)


def _detector(found, clock=None):
  return ScreenDetector(FakeMatcher(found),
                        now=clock or (lambda: 1.0))


def _real_detector(tmp_path):
  """Build a real matcher/detector over the committed popup templates.

  Args:
    tmp_path: Temporary directory used as the template library root.

  Returns:
    tuple: The detector, matcher and library built from real PNG templates.
  """
  for name in F23_TEMPLATES:
    shutil.copy(PROJECT_ROOT / 'templates' / f'{name}.png',
                tmp_path / f'{name}.png')
  manifest = tmp_path / 'manifest.json'
  manifest.write_text(json.dumps({
      'version': 1,
      'templates': {name: f'{name}.png' for name in F23_TEMPLATES},
  }), encoding='utf-8')
  library = TemplateLibrary(str(tmp_path))
  library.load(str(manifest))
  matcher = TemplateMatcher(library)
  return ScreenDetector(matcher, now=lambda: 1.0), matcher, library


def _committed_detector():
  """Build a real detector from the full committed template directory.

  Returns:
    tuple: The detector, matcher and library built from all committed PNGs.
  """
  library = TemplateLibrary(str(PROJECT_ROOT / 'templates'))
  library.load(str(MANIFEST_PATH))
  matcher = TemplateMatcher(library)
  return ScreenDetector(matcher, now=lambda: 1.0), matcher, library


def test_state_rules_follow_the_locked_order():
  """STATE_RULES is ordered GIFT_DIALOG-first and HOME-last."""
  states = [state for state, _ in ScreenDetector.STATE_RULES]
  assert states == [
      GameState.GIFT_DIALOG, GameState.POPUP, GameState.DIALOG,
      GameState.TUTORIAL, GameState.BAG_DIALOG, GameState.RECIPE_DIALOG,
      GameState.TAP_TO_START, GameState.CRAFTING, GameState.CATALOG,
      GameState.HOME]


def test_popup_beats_home_when_both_match():
  """A first-rule POPUP wins over a later HOME match."""
  detector = _detector({'close_dialog': [_match('close_dialog')],
                        'home_marker': [_match('home_marker')]})
  assert detector.detect(_screen()).state is GameState.POPUP


def test_every_banner_classifies_as_popup_or_dialog():
  """Each captured popup banner drives a dismissible modal state."""
  assert len(BANNER_MARKERS) == 6
  for name in BANNER_MARKERS:
    detector = _detector({name: [_match(name)]})
    assert detector.detect(_screen()).state in (
        GameState.POPUP, GameState.DIALOG)


def test_close_dialog_is_the_only_popup_fallback():
  """The generic close button is the POPUP fallback, not the variants."""
  assert 'close_dialog' in POPUP_MARKERS
  for name in ('close_dialog_2', 'close_dialog_3',
               'close_dialog_4', 'close_dialog_5'):
    assert name not in POPUP_MARKERS


def test_close_dialog_variants_are_not_state_markers():
  """Floating close X variants never drive state away from CRAFTING."""
  for name in ('close_dialog_2', 'close_dialog_3',
               'close_dialog_4', 'close_dialog_5'):
    detector = _detector({name: [_match(name)],
                          'crafting_marker': [_match('crafting_marker')]})
    assert detector.detect(_screen()).state is GameState.CRAFTING


def test_craft_shelf_with_close_x_stays_crafting():
  """A craft shelf whose close X matches a variant is CRAFTING, not DIALOG."""
  found = {'close_dialog_2': [_match('close_dialog_2', score=0.978)],
           'crafting_marker': [_match('crafting_marker')],
           'home_marker': [_match('home_marker')]}
  assert _detector(found).detect(_screen()).state is GameState.CRAFTING


def test_gift_dialog_detected():
  """The banner and Collect button together identify the gift dialog."""
  detector = _detector({
      'banner_coins': [_match('banner_coins')],
      'gift_collect': [_match('gift_collect')],
  })
  assert detector.detect(_screen()).state is GameState.GIFT_DIALOG


def test_gift_collect_alone_is_not_gift_dialog():
  """A lone Collect match must not misclassify another COINS promo."""
  detector = _detector({'gift_collect': [_match('gift_collect')]})
  assert detector.detect(_screen()).state is not GameState.GIFT_DIALOG


def test_banner_coins_alone_is_popup_not_gift_dialog():
  """A COINS banner without a Collect control stays a normal POPUP."""
  detector = _detector({'banner_coins': [_match('banner_coins')]})
  assert detector.detect(_screen()).state is GameState.POPUP


def test_gift_collect_score_override_is_pinned_in_code():
  """gift_collect keeps its pinned 0.45 override; craft_ready_check 0.97."""
  assert TEMPLATE_SCORE_OVERRIDES['gift_collect'] == 0.45
  assert TEMPLATE_SCORE_OVERRIDES['craft_ready_check'] == 0.97


def test_real_coins_popup_is_gift_dialog_through_real_rules(tmp_path):
  """The committed real COINS screenshot classifies GIFT_DIALOG."""
  detector, _, _ = _real_detector(tmp_path)
  image = cv2.imread(str(FIXTURE_PATH), cv2.IMREAD_COLOR)
  assert image is not None
  detection = detector.detect(Screen(image))
  assert detection.state is GameState.GIFT_DIALOG
  collect = Navigator(detector).action_for_template(
      detection, 'gift_collect')
  assert collect is not None
  assert collect.reason == 'gift_collect'


@pytest.mark.parametrize('banner', [
    'banner_flash_sale', 'banner_vip', 'banner_club_token',
    'banner_harvest',
])
def test_real_non_gift_banner_is_dismissed_without_collect(
    tmp_path, banner):
  """A real non-COINS banner is a POPUP/DIALOG with no Collect tap."""
  detector, _, library = _real_detector(tmp_path)
  image = np.full((320, 1000, 3), 200, dtype=np.uint8)
  template = library.get(banner)
  height, width = template.shape[:2]
  image[20:20 + height, 20:20 + width] = template
  detection = detector.detect(Screen(image))
  assert detection.state in (GameState.POPUP, GameState.DIALOG)
  action = Navigator(detector).choose(detection, Memory(), None)
  assert action.reason != 'gift_collect'


def test_real_coins_promo_without_collect_is_popup(tmp_path):
  """A real COINS banner with its Collect control blanked stays POPUP."""
  detector, _, _ = _real_detector(tmp_path)
  image = cv2.imread(str(FIXTURE_PATH), cv2.IMREAD_COLOR)
  assert image is not None
  image[1500:1700, 350:750] = 200
  detection = detector.detect(Screen(image))
  assert detection.state in (GameState.POPUP, GameState.DIALOG)
  action = Navigator(detector).choose(detection, Memory(), None)
  assert action.reason != 'gift_collect'


def test_each_remaining_state_detected():
  """Each marker template maps to its locked game state."""
  cases = [
      ('tutorial_skip', GameState.TUTORIAL),
      ('inventory_marker', GameState.BAG_DIALOG),
      ('recipe_marker', GameState.RECIPE_DIALOG),
      ('tap_to_start', GameState.TAP_TO_START),
      ('crafting_marker', GameState.CRAFTING),
      ('catalog_marker', GameState.CATALOG),
      ('home_marker', GameState.HOME),
  ]
  for name, state in cases:
    detector = _detector({name: [_match(name)]})
    assert detector.detect(_screen()).state is state


def test_state_rules_reference_only_real_templates():
  """Every detection template name exists in the real manifest."""
  manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
  available = set(manifest['templates'])
  referenced = {
      name
      for _, names in ScreenDetector.STATE_RULES
      for name in names
  }
  referenced.add(LOADING_TEXT)
  missing = referenced - available
  assert not missing, f'missing templates: {sorted(missing)}'


def test_bag_marker_alone_is_not_bag_dialog():
  """A lone floating Bag marker never means the inventory is open."""
  detector = _detector({'bag_marker': [_match('bag_marker')]})
  assert detector.detect(_screen()).state in (
      GameState.CRAFTING, GameState.UNKNOWN)


def test_crafting_shelf_bag_marker_stays_crafting():
  """The floating Bag button on the crafting shelf keeps CRAFTING."""
  detector = _detector({'bag_marker': [_match('bag_marker')],
                        'crafting_marker': [_match('crafting_marker')]})
  assert detector.detect(_screen()).state is GameState.CRAFTING


def test_loading_text_triggers_loading():
  """A loading_text match yields LOADING."""
  detector = _detector({'loading_text': [_match('loading_text')]})
  assert detector.detect(_screen()).state is GameState.LOADING


def test_dark_screen_triggers_loading():
  """A near-black screen is recognised as LOADING."""
  detector = _detector({})
  assert detector.detect(_screen(0)).state is GameState.LOADING


def test_unknown_when_nothing_matches():
  """A bright screen with no matches is UNKNOWN."""
  detector = _detector({})
  assert detector.detect(_screen(255)).state is GameState.UNKNOWN


def test_timestamp_comes_from_injected_clock():
  """Detection.timestamp uses the injected now callable."""
  detector = _detector({}, clock=lambda: 99.5)
  assert detector.detect(_screen()).timestamp == 99.5


def test_matches_come_from_interactive_matches():
  """Detection.matches equals the interactive-match producer output."""
  found = {'home_marker': [_match('home_marker')],
           'bag_marker': [_match('bag_marker')]}
  detector = _detector(found)
  screen = _screen()
  detection = detector.detect(screen)
  assert detection.matches == tuple(detector.interactive_matches(screen))


def test_interactive_matches_filters_loading_text():
  """The loading-text template is not an interactive match."""
  found = {'loading_text': [_match('loading_text')],
           'home_marker': [_match('home_marker')]}
  detector = _detector(found)
  names = [m.name for m in detector.interactive_matches(_screen())]
  assert names == ['home_marker']


def test_is_loading_dark_screen():
  """is_loading reports True for a black screen."""
  assert is_loading(_screen(0))


def test_is_loading_bright_screen():
  """is_loading reports False for a bright screen."""
  assert not is_loading(_screen(255))


def test_fresh_fruit_banner_is_a_state_marker():
  """The real FRESH FRUIT ribbon drives a dismissible modal state."""
  assert FRESH_FRUIT_BANNER in BANNER_MARKERS
  detector = _detector({FRESH_FRUIT_BANNER: [_match(FRESH_FRUIT_BANNER)]})
  assert detector.detect(_screen()).state in (
      GameState.POPUP, GameState.DIALOG)


def test_manifest_registers_fresh_fruit_banner():
  """The manifest keeps its 55 prior templates plus the new banner."""
  manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
  assert manifest['templates'][FRESH_FRUIT_BANNER] == (
      f'{FRESH_FRUIT_BANNER}.png')
  assert len(manifest['templates']) == 56
  assert (PROJECT_ROOT / 'templates' /
          f'{FRESH_FRUIT_BANNER}.png').is_file()


def test_real_fresh_fruit_template_scores_high_on_source():
  """The committed banner template matches the real popup at >= 0.85."""
  _, _, library = _committed_detector()
  image = cv2.imread(str(FRESH_FRUIT_FIXTURE), cv2.IMREAD_COLOR)
  assert image is not None
  template = library.get(FRESH_FRUIT_BANNER)
  scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
  assert float(scores.max()) >= 0.85


def test_real_fresh_fruit_popup_is_dismissed_through_real_rules():
  """The real FRESH FRUIT screenshot is a BACK-dismissed POPUP/DIALOG."""
  detector, _, _ = _committed_detector()
  image = cv2.imread(str(FRESH_FRUIT_FIXTURE), cv2.IMREAD_COLOR)
  assert image is not None
  detection = detector.detect(Screen(image))
  assert detection.state in (GameState.POPUP, GameState.DIALOG)
  action = Navigator(detector).choose(detection, Memory(), None)
  assert action.kind is ActionKind.BACK
  assert action.reason == 'dismiss'


def test_fresh_fruit_banner_absent_from_normal_shelf():
  """A craft shelf with the ribbon blanked is not a FRESH FRUIT POPUP."""
  detector, matcher, _ = _committed_detector()
  image = cv2.imread(str(FRESH_FRUIT_FIXTURE), cv2.IMREAD_COLOR)
  assert image is not None
  image[620:800, 100:960] = 200
  screen = Screen(image)
  assert not matcher.match(screen, FRESH_FRUIT_BANNER)
  detection = detector.detect(screen)
  assert detection.state not in (GameState.POPUP, GameState.DIALOG)
