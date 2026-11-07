#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro settings for the forecast ML framework.

Hooks are ``ForecastHooks`` (see ``HOOKS``); do not add
business logic to this module.

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

``CONF_SOURCE`` is resolved by :func:`_resolve_conf_source`
(V10): an ``importlib.resources``-packaged ``conf`` wins
when the installed wheel ships one; otherwise the
checkout layout (``<project>/conf``) is used. An
installed layout without a packaged ``conf`` has no
checkout to fall back to, so the checkout path is still
returned and the CLI reports "project not found" (exit 2)
rather than crashing here.
'''

import logging

from pathlib import Path

LOG = logging.getLogger(__name__)

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
          'oc.* resolvers are blocked in Kedro config loader')

    def _merge_and_resolve(self, key, config_per_file):
      '''Reject ``oc.*`` in raw configs, then resolve.'''
      from forecast_ml_framework.core.config_schema import (  # pylint: disable=import-outside-toplevel
        _blocked_oc_resolvers,
        assert_no_cross_key_oc,
        reject_oc_in_resolved,
      )
      for _cfg in config_per_file.values():
        self._reject_oc(
          OmegaConf.to_container(_cfg, resolve=False))
      # Fragments split across keys or files recombine only
      # during resolve(); check the merged unresolved
      # container, resolve with oc.* resolvers blocked, then
      # re-scan the fully resolved result.
      _cfgs = list(config_per_file.values())
      if _cfgs:
        _merged = OmegaConf.unsafe_merge(*_cfgs)
        assert_no_cross_key_oc(
          OmegaConf.to_container(_merged, resolve=False))
      with _blocked_oc_resolvers():
        _result = super()._merge_and_resolve(key, config_per_file)
      reject_oc_in_resolved(_result)
      return _result

    @staticmethod
    def _resolve_environment_variables(config):
      '''Refuse ``oc.env`` resolution for credentials.'''
      from forecast_ml_framework.core.config_schema import (  # pylint: disable=import-outside-toplevel
        assert_no_cross_key_oc,
      )
      plain = OmegaConf.to_container(config, resolve=False)
      RestrictedOmegaConfigLoader._reject_oc(plain)
      assert_no_cross_key_oc(plain)

  CONFIG_LOADER_CLASS = RestrictedOmegaConfigLoader
except ImportError:  # pragma: no cover - kedro optional at import
  CONFIG_LOADER_CLASS = None

try:
  from forecast_ml_framework.hooks import ForecastHooks

  HOOKS = (ForecastHooks(),)
except ImportError as exc:  # pragma: no cover - hooks optional at import
  LOG.warning(
      'ForecastHooks unavailable code=%s type=%s',
      'hooks_unavailable', type(exc).__name__)
  HOOKS = ()


def _resolve_conf_source():
  '''Resolve the ``conf`` directory for this checkout/install.

  Returns:
    Absolute path string: the ``importlib.resources``-
    packaged ``conf`` (``<package>/conf``) when present,
    else the checkout layout ``<project>/conf`` (the
    ``parents[2]`` fallback, kept for source checkouts).
  '''
  try:
    import importlib.resources
    packaged = (
        importlib.resources.files('forecast_ml_framework')
        .joinpath('conf'))
    if Path(str(packaged)).is_dir():
      return str(packaged)
  except (ImportError, OSError, TypeError, ValueError,
          AttributeError) as exc:
    LOG.debug('_resolve_conf_source: packaged conf skipped (%s)', exc)
  return str(Path(__file__).resolve().parents[2] / 'conf')


CONF_SOURCE = _resolve_conf_source()
__all__ = ['CONFIG_LOADER_CLASS', 'CONF_SOURCE', 'HOOKS']
