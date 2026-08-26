#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pytest path bootstrap: expose src/knee packages to the test runner."""

import os
import sys

SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC))
