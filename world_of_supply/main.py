#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Entry-point shim delegating to the package CLI.

Allows running the project from a source checkout without installation by
putting ``src`` on ``sys.path`` first.
'''

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / 'src'
if str(_SRC) not in sys.path:
  sys.path.insert(0, str(_SRC))

from world_of_supply.cli import main  # noqa: E402

if __name__ == '__main__':
  sys.exit(main())
