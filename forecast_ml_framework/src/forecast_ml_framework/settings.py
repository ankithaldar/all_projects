#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro settings stub for the forecast ML framework.

Hooks are empty in the scaffold phase. Future phases may add
hooks here; do not add business logic to this module.

``CONFIG_LOADER_CLASS`` is
``RestrictedOmegaConfigLoader``, a Kedro
``OmegaConfigLoader`` subclass that rejects any
``${oc.*}`` interpolation (including whitespace/case
variants such as ``${ oc.env:...}``) before OmegaConf
resolves it, so the Kedro path cannot read the process
environment via ``oc.env`` / ``oc.create``. This routes
the Kedro loading path through the same sanitizer
principle as ``core.config_schema.load_config`` /
``core.config_loader.ConfigLoader``. Non-Kedro callers
must use ``load_config`` and must never use the stock
``OmegaConfigLoader`` directly.
'''

try:
  import re

  from kedro.config import OmegaConfigLoader
  from omegaconf import OmegaConf

  _OC_PATTERN = re.compile(r'\$\{\s*oc\s*\.', re.IGNORECASE)

  class RestrictedOmegaConfigLoader(OmegaConfigLoader):
    '''OmegaConfigLoader rejecting ``oc.*`` interpolations.

    Checks every raw (unresolved) config node for
    ``${oc...}`` including whitespace/case variants and
    raises ``ValueError`` before resolution, so
    ``oc.env`` can never leak a process-environment
    secret through the Kedro path. The credentials
    ``oc.env`` re-registration in the stock loader is
    disabled for the same reason.
    '''

    @staticmethod
    def _reject_oc(node):
      '''Raise ValueError if ``node`` contains ``oc.*``.'''
      if isinstance(node, dict):
        for _value in node.values():
          RestrictedOmegaConfigLoader._reject_oc(_value)
      elif isinstance(node, (list, tuple)):
        for _value in node:
          RestrictedOmegaConfigLoader._reject_oc(_value)
      elif isinstance(node, str) and _OC_PATTERN.search(node):
        raise ValueError(
          'oc.* resolvers are blocked in Kedro config '
          'loader: ' + node)

    def _merge_and_resolve(self, key, config_per_file):
      '''Reject ``oc.*`` in raw configs, then resolve.'''
      for _cfg in config_per_file.values():
        self._reject_oc(
          OmegaConf.to_container(_cfg, resolve=False))
      return super()._merge_and_resolve(key, config_per_file)

    @staticmethod
    def _resolve_environment_variables(config):
      '''Refuse ``oc.env`` resolution for credentials.'''
      plain = OmegaConf.to_container(config, resolve=False)
      RestrictedOmegaConfigLoader._reject_oc(plain)

  CONFIG_LOADER_CLASS = RestrictedOmegaConfigLoader
except ImportError:  # pragma: no cover - kedro optional at import
  CONFIG_LOADER_CLASS = None

CONF_SOURCE = 'conf'
HOOKS = ()
__all__ = ['CONFIG_LOADER_CLASS', 'CONF_SOURCE', 'HOOKS']
