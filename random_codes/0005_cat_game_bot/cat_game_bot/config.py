#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Frozen runtime configuration and shared authoritative path confinement."""

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path

from cat_game_bot.adb import validate_package, validate_serial
from cat_game_bot.errors import ConfigError
from cat_game_bot.models import Rect

__all__ = (
  'CANONICAL_IDS', 'Config', 'CRAFTABLE_IDS', 'OFFSET_NAMES', 'PROJECT_ROOT',
  'SCHEDULE_PATH', '_json_default', 'load_regions', 'parse_args',
  '_resolve_path')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fixed, trusted location of the user-mandated best-schedule CSV. It is a
# module constant rather than a CLI flag and intentionally bypasses
# ``_resolve_path``: crafting.load_schedule validates it with an 8 MiB cap
# and a strict schema instead of project-root confinement.
SCHEDULE_PATH: str = (
  '/run/media/helder/Code_Dump/all_projects_all_branches/'
  'crazy_ideas_main/random_codes/0004_cat_game_neat/data/'
  'best-schedule.csv')

_PATH_RULES = {
  'template_dir': 'templates',
  'log_dir': 'logs',
  'tesseract_cmd': '.tools',
  'targets_path': 'data',
  'tree_path': 'data',
  'ocr_regions_path': 'data',
  'db_path': 'logs',
}

CANONICAL_IDS = (
  'cotton', 'tree', 'rock', 'quartz',
  'string', 'wood', 'metal', 'bronze', 'amethyst', 'orb', 'ribbon',
  'needles', 'sparkles', 'silver', 'gold', 'pendant', 'water',
  'necklace', 'fire', 'waterstone', 'firestone', 'elementstone',
  'artifact')
OFFSET_NAMES = frozenset(f'item_count.{item}' for item in CANONICAL_IDS)
_BASE_IDS = frozenset(('cotton', 'tree', 'rock', 'quartz'))
# The 19 craftable ids in canonical tree order; this is the exact column
# order required of the fixed schedule header.
CRAFTABLE_IDS = tuple(item for item in CANONICAL_IDS
                      if item not in _BASE_IDS)
_REGION_NAMES = frozenset(('coins', 'gift_timer'))
_TOP_KEYS = frozenset(('version', 'regions', 'offsets'))
_REGION_KEYS = frozenset(('x', 'y', 'w', 'h'))
_OFFSET_KEYS = frozenset(('dx', 'dy', 'w', 'h'))


def _json_default(value: object) -> object:
  """Convert enum, path and dataclass values for JSON serialisation.

  This is the single canonical default used by both the session's config
  dump and the database's canonical payload encoder.

  Args:
    value: Value not natively handled by json.

  Returns:
    A JSON-native representation of the value.
  """
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, Path):
    return str(value)
  if is_dataclass(value):
    return asdict(value)
  return str(value)


@dataclass(frozen=True)
class Config:
  """Immutable validated configuration for one bot session."""

  serial: str = 'emulator-5554'
  package: str = 'com.minogames.cats.client'
  duration_seconds: float = 300.0
  max_actions: int = 500
  bag_visit_interval: int = 10
  template_dir: Path = field(
    default_factory=lambda: PROJECT_ROOT / 'templates')
  threshold: float = 0.8
  log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'logs')
  poll_delay: float = 1.0
  settle_timeout: float = 12.0
  dry_run: bool = True
  no_progress_limit: int = 15
  error_limit: int = 5
  tesseract_cmd: Path = field(
    default_factory=lambda: PROJECT_ROOT / '.tools' / 'bin' / 'tesseract')
  targets_path: Path = field(
    default_factory=lambda: PROJECT_ROOT / 'data' / 'initial-state.json')
  tree_path: Path = field(
    default_factory=lambda: PROJECT_ROOT / 'data' / 'crafting_tree.yaml')
  ocr_regions_path: Path = field(
    default_factory=lambda: PROJECT_ROOT / 'data' / 'ocr-regions.json')
  db_path: Path = field(
    default_factory=lambda: PROJECT_ROOT / 'logs' / 'bot.sqlite3')
  ocr_lang: str = 'eng'


def _resolve_path(p: str | Path, allowed_root: str | Path) -> Path:
  """Confine a path to an allowed root without using assertions.

  Args:
    p: The candidate path.
    allowed_root: The root the resolved path must remain under.

  Returns:
    The resolved absolute path.

  Raises:
    ConfigError: If the path is a symlink or escapes the allowed root.
  """
  candidate = Path(p)
  if candidate.is_symlink():
    raise ConfigError(f'symlink path rejected: {p}')
  resolved = candidate.resolve()
  if not resolved.is_relative_to(Path(allowed_root).resolve()):
    raise ConfigError(f'path outside allowed root: {p}')
  return resolved


def _bounded(value: float, name: str, low: float, high: float) -> float:
  """Return a numeric value after enforcing an inclusive range.

  Args:
    value: The value to check.
    name: Field name used in the error message.
    low: Inclusive lower bound.
    high: Inclusive upper bound.

  Returns:
    The unchanged value when it lies within the range.

  Raises:
    ConfigError: If the value lies outside the range.
  """
  if not low <= value <= high:
    raise ConfigError(f'{name} out of range [{low}, {high}]')
  return value


def _strict_int(value: object, label: str) -> int:
  """Return a plain int, rejecting bools and all other types.

  Args:
    value: Value to inspect.
    label: Field label used in the error message.

  Returns:
    The validated integer.

  Raises:
    ConfigError: If the value is not a plain int.
  """
  if isinstance(value, bool) or not isinstance(value, int):
    raise ConfigError(f'{label} must be an int')
  return value


def _exact_keys(mapping: object, expected: frozenset[str], label: str) -> None:
  """Require a mapping to have exactly the expected key set.

  Args:
    mapping: Mapping to inspect.
    expected: The exact set of allowed keys.
    label: Mapping label used in the error message.

  Raises:
    ConfigError: If keys are missing or unexpected.
  """
  if not isinstance(mapping, dict) or set(mapping) != expected:
    raise ConfigError(f'invalid keys for {label}')


def _parse_region(name: str, box: object) -> Rect:
  """Validate one absolute OCR region box and build a Rect.

  Args:
    name: Region name used in error messages.
    box: Mapping with exactly x/y/w/h integer values.

  Returns:
    The validated region rectangle.

  Raises:
    ConfigError: On any key, type or bound violation.
  """
  _exact_keys(box, _REGION_KEYS, f'region {name}')
  x = _strict_int(box['x'], f'{name}.x')
  y = _strict_int(box['y'], f'{name}.y')
  width = _strict_int(box['w'], f'{name}.w')
  height = _strict_int(box['h'], f'{name}.h')
  if x < 0 or y < 0:
    raise ConfigError(f'{name} has negative coordinates')
  if width <= 0 or height <= 0:
    raise ConfigError(f'{name} has a non-positive extent')
  if x + width > 1080 or y + height > 2400:
    raise ConfigError(f'{name} exceeds screen bounds')
  if width * height > 2592000:
    raise ConfigError(f'{name} exceeds the area cap')
  return Rect(x, y, width, height)


def _parse_offset(name: str, box: object) -> tuple[int, int, int, int]:
  """Validate one relative count-offset box.

  Args:
    name: Offset name used in error messages.
    box: Mapping with exactly dx/dy/w/h integer values.

  Returns:
    The validated relative box as ``(dx, dy, w, h)``.

  Raises:
    ConfigError: On any key, type or bound violation.
  """
  _exact_keys(box, _OFFSET_KEYS, f'offset {name}')
  dx = _strict_int(box['dx'], f'{name}.dx')
  dy = _strict_int(box['dy'], f'{name}.dy')
  width = _strict_int(box['w'], f'{name}.w')
  height = _strict_int(box['h'], f'{name}.h')
  if dx < 0 or dy < 0:
    raise ConfigError(f'{name} has negative coordinates')
  if not 0 < width <= 200 or not 0 < height <= 100:
    raise ConfigError(f'{name} has an out-of-range extent')
  return (dx, dy, width, height)


def load_regions(path: str) -> tuple[dict, dict]:
  """Load and strictly validate the ocr-regions JSON document.

  Args:
    path: Path to ocr-regions.json, confined by ``parse_args``.

  Returns:
    A ``(regions, offsets)`` pair: regions maps name to Rect, offsets
    maps name to a four-integer relative box.

  Raises:
    ConfigError: If the file or any part of its schema is invalid.
  """
  try:
    with open(path, 'r', encoding='utf-8') as handle:
      data = json.load(handle)
  except (OSError, ValueError) as exc:
    raise ConfigError(f'invalid ocr-regions file: {exc}') from exc
  if not isinstance(data, dict):
    raise ConfigError('ocr-regions file must be a JSON object')
  _exact_keys(data, _TOP_KEYS, 'ocr-regions document')
  version = data['version']
  if (isinstance(version, bool) or not isinstance(version, int)
      or version != 1):
    raise ConfigError('unsupported ocr-regions version')
  raw_regions = data['regions']
  raw_offsets = data['offsets']
  _exact_keys(raw_regions, _REGION_NAMES, 'regions map')
  _exact_keys(raw_offsets, OFFSET_NAMES, 'offsets map')
  regions = {name: _parse_region(name, raw_regions[name])
             for name in _REGION_NAMES}
  offsets = {name: _parse_offset(name, raw_offsets[name])
             for name in OFFSET_NAMES}
  return regions, offsets


def parse_args(argv: Sequence[str] | None = None) -> Config:
  """Parse, validate and confine command-line configuration.

  Args:
    argv: Argument tokens; defaults to the process command line.

  Returns:
    A validated frozen Config with the log directory created.

  Raises:
    ConfigError: On invalid values, unsafe paths or an uncreatable log dir.
  """
  root = PROJECT_ROOT
  parser = argparse.ArgumentParser(prog='cat_game_bot')
  parser.add_argument('--serial', required=True)
  parser.add_argument('--package', default='com.minogames.cats.client')
  parser.add_argument('--duration', type=float, default=300.0,
                      dest='duration_seconds')
  parser.add_argument('--max-actions', type=int, default=500)
  parser.add_argument('--bag-visit-interval', type=int, default=10)
  parser.add_argument('--templates', default=str(root / 'templates'),
                      dest='template_dir')
  parser.add_argument('--threshold', type=float, default=0.8)
  parser.add_argument('--log-dir', default=str(root / 'logs'))
  parser.add_argument('--poll-delay', type=float, default=1.0)
  parser.add_argument('--settle-timeout', type=float, default=12.0)
  parser.add_argument('--live', dest='dry_run', action='store_false',
                      default=True)
  parser.add_argument('--no-progress-limit', type=int, default=15)
  parser.add_argument('--error-limit', type=int, default=5)
  parser.add_argument('--tesseract-cmd',
                      default=str(root / '.tools' / 'bin' / 'tesseract'))
  parser.add_argument('--targets',
                      default=str(root / 'data' / 'initial-state.json'),
                      dest='targets_path')
  parser.add_argument('--tree',
                      default=str(root / 'data' / 'crafting_tree.yaml'),
                      dest='tree_path')
  parser.add_argument('--ocr-regions',
                      default=str(root / 'data' / 'ocr-regions.json'),
                      dest='ocr_regions_path')
  parser.add_argument('--db',
                      default=str(root / 'logs' / 'bot.sqlite3'),
                      dest='db_path')
  parser.add_argument('--ocr-lang', default='eng')
  ns = parser.parse_args(argv)

  serial = validate_serial(ns.serial)
  package = validate_package(ns.package)
  duration = _bounded(ns.duration_seconds, 'duration_seconds', 1.0, 3600.0)
  max_actions = _bounded(ns.max_actions, 'max_actions', 1, 10000)
  bag_interval = _bounded(ns.bag_visit_interval, 'bag_visit_interval',
                          1, 100)
  threshold = _bounded(ns.threshold, 'threshold', 0.5, 1.0)
  poll_delay = _bounded(ns.poll_delay, 'poll_delay', 0.2, 5.0)
  settle_timeout = _bounded(ns.settle_timeout, 'settle_timeout', 2.0, 60.0)
  no_progress = _bounded(ns.no_progress_limit, 'no_progress_limit', 1, 100)
  error_limit = _bounded(ns.error_limit, 'error_limit', 1, 20)

  paths = {
    name: _resolve_path(getattr(ns, name), root / subdir)
    for name, subdir in _PATH_RULES.items()
  }
  log_dir = paths['log_dir']
  try:
    log_dir.mkdir(parents=True, exist_ok=True)
  except OSError as exc:
    raise ConfigError(f'cannot create log dir: {log_dir}') from exc
  try:
    os.chmod(log_dir, 0o700)
  except OSError as exc:
    raise ConfigError(f'cannot secure log dir: {log_dir}') from exc

  return Config(
    serial=serial,
    package=package,
    duration_seconds=duration,
    max_actions=max_actions,
    bag_visit_interval=bag_interval,
    template_dir=paths['template_dir'],
    threshold=threshold,
    log_dir=paths['log_dir'],
    poll_delay=poll_delay,
    settle_timeout=settle_timeout,
    dry_run=ns.dry_run,
    no_progress_limit=no_progress,
    error_limit=error_limit,
    tesseract_cmd=paths['tesseract_cmd'],
    targets_path=paths['targets_path'],
    tree_path=paths['tree_path'],
    ocr_regions_path=paths['ocr_regions_path'],
    db_path=paths['db_path'],
    ocr_lang=ns.ocr_lang,
  )
