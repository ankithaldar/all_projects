#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Canonical command-line entry point for the Cat Game bot."""

# '__main__' is the mandated canonical CLI module name (plan v11 section 4).
# pylint: disable=invalid-name

from pathlib import Path

from cat_game_bot.adb import AdbClient
from cat_game_bot.config import SCHEDULE_PATH, load_regions, parse_args
from cat_game_bot.crafting import CraftPlanner, load_tree
from cat_game_bot.db import InteractionLog
from cat_game_bot.detection import ScreenDetector
from cat_game_bot.errors import DbError
from cat_game_bot.gift import GiftCollector
from cat_game_bot.inventory import InventoryReader
from cat_game_bot.navigator import Navigator
from cat_game_bot.ocr import TesseractReader
from cat_game_bot.session import Session
from cat_game_bot.targets import load_targets
from cat_game_bot.vision import TemplateLibrary, TemplateMatcher


def _build_session(cfg, db):
  """Construct the injected collaborator graph for one run.

  Args:
      cfg: Parsed frozen configuration.
      db: Open interaction log.

  Returns:
      Session: Fully wired session ready to run.
  """
  adb = AdbClient(cfg.serial)
  library = TemplateLibrary(str(cfg.template_dir))
  library.load(str(Path(cfg.template_dir) / 'manifest.json'))
  matcher = TemplateMatcher(library, cfg.threshold)
  detector = ScreenDetector(matcher)
  ocr = TesseractReader(str(cfg.tesseract_cmd), cfg.ocr_lang)
  regions, offsets = load_regions(str(cfg.ocr_regions_path))
  inventory = InventoryReader(ocr, matcher, offsets, library)
  spec = load_targets(str(cfg.targets_path))
  tree = load_tree(str(cfg.tree_path), str(Path(cfg.tree_path).parent))
  planner = CraftPlanner(tree, spec, {}, schedule_path=SCHEDULE_PATH)
  gift = GiftCollector(ocr, regions)
  navigator = Navigator(detector)
  return Session(adb, detector, navigator, ocr, inventory, planner,
                 gift, db, cfg, spec)


def _run(argv):
  """Parse arguments, run one session and always close the log.

  Args:
      argv: Optional argument sequence.

  Returns:
      SessionStats: Statistics from the run.
  """
  cfg = parse_args(argv)
  db = InteractionLog(str(cfg.db_path))
  try:
    return _build_session(cfg, db).run()
  finally:
    db.close()


def main(argv=None):
  """Map the terminal result to a process exit code.

  Args:
      argv: Optional argument sequence.

  Returns:
      int: 0 on GOAL_MET, 1 on any other result, 2 on a DbError.
  """
  try:
    stats = _run(argv)
  except DbError:
    return 2
  if stats.result == 'GOAL_MET':
    return 0
  return 1


if __name__ == '__main__':
  raise SystemExit(main())
