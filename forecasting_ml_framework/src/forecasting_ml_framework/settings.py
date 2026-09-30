#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Project settings: how the framework's own project is wired.

The ``globals`` configuration group is a framework-specific invention — stock
Kedro has no such namespace. It provides a second, higher-priority parameter
channel, letting a model instance override a column list without a bespoke
per-model parameters document. Resolution is achieved not by the loader but by
the bootstrap entry point rewriting ``${KEY`` into ``${globals:KEY`` on disk
before Kedro reads the file, so the ``globals`` pattern here is a genuine
``${globals:...}`` resolver target rather than a merged namespace.
"""

from __future__ import annotations

from pathlib import Path

from kedro.config import OmegaConfigLoader

from forecasting_ml_framework.hooks import ModelTrackingHooks

#: The Spark-aware Kedro context. It delegates session construction entirely to
#: the single platform builder, so the connector coordinates and the credential
#: refresh have exactly one definition however the session came into being --
#: which is what lets the framework be exercised outside a cluster and hosted on
#: one without the two paths diverging.
from forecasting_ml_framework.platform.context import ForecastingSparkContext

#: The hook manager. Hooks execute last-in-first-out within a manager.
HOOKS = (ModelTrackingHooks(),)

CONF_SOURCE = 'conf'

#: The configuration-loader arguments. These are *overridden*, not merged, with
#: Kedro's defaults, so the patterns below are the complete set.
CONFIG_LOADER_ARGS: dict = {
  'base_env': 'base',
  'default_run_env': 'local',
  'config_patterns': {
    'spark': ['spark.yml'],
    'parameters': ['parameters*', 'parameters*/**', '**/parameters*'],
    'catalog': ['catalog*', 'catalog*/**', '**/catalog*'],
    'globals': ['globals.yml'],
  },
}

CONFIG_LOADER_CLASS = OmegaConfigLoader

CONTEXT_CLASS = ForecastingSparkContext

# Paths — useful for the configuration rewriter and for deployment scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = PROJECT_ROOT / CONF_SOURCE
