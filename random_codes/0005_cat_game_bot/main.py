#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Thin wrapper delegating to the canonical package CLI."""

from cat_game_bot.__main__ import main


if __name__ == '__main__':
  raise SystemExit(main())
