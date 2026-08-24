#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Pytest bootstrap: make the src layout importable without installation.'''

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
  sys.path.insert(0, str(SRC))
