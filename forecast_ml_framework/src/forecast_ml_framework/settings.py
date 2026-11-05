#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro settings stub for the forecast ML framework.

Hooks are empty in the scaffold phase. Future phases may add
hooks here; do not add business logic to this module.
'''

try:
  from kedro.config import OmegaConfigLoader

  CONFIG_LOADER_CLASS = OmegaConfigLoader
except ImportError:  # pragma: no cover - kedro optional at import
  CONFIG_LOADER_CLASS = None

CONF_SOURCE = 'conf'
HOOKS = ()
__all__ = ['CONFIG_LOADER_CLASS', 'CONF_SOURCE', 'HOOKS']
