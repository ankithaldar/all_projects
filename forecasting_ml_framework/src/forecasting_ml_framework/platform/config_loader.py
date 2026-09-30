#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration loader with a ``runtime`` OmegaConf resolver.

It provides two things:

* a ``runtime`` resolver is registered, so a configuration document can address a
  runtime parameter by name without the bootstrap rewriter having to rewrite every
  occurrence to ``${runtime_params:NAME}``;
* the loader reads the base environment and the config patterns from the
  arguments it was actually constructed with, rather than from attributes that do
  not exist.

The bootstrap rewriter remains the mechanism for the three-tier precedence,
because it decides *which channel* each key resolves from. This resolver is the
convenience layer on top of it.
"""

from __future__ import annotations

from typing import Any

from kedro.config import OmegaConfigLoader
from omegaconf import OmegaConf

from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Prefix under which a runtime parameter may also be supplied directly through
#: the process environment, for a run that is launched without the bootstrap.
ENV_PREFIX = 'FORECASTING_ML_PARAM_'


class RuntimeResolverConfigLoader(OmegaConfigLoader):
  """An OmegaConf loader that exposes runtime parameters as ``${runtime:NAME}``.

  Args:
    *args: Positional arguments forwarded to the parent loader.
    conf_source: The configuration root.
    base_env: The base environment directory.
    default_run_env: The environment used when none is requested.
    config_patterns: The per-group glob patterns.
    **kwargs: Additional keyword arguments forwarded to the parent loader.
  """

  def __init__(
    self,
    *args: Any,
    conf_source: str = 'conf',
    base_env: str = 'base',
    default_run_env: str = 'local',
    config_patterns: dict[str, list[str]] | None = None,
    **kwargs: Any,
  ) -> None:
    """Build the loader and register the resolver.

    Args:
      *args: Positional arguments forwarded to the parent loader.
      conf_source: The configuration root.
      base_env: The base environment directory.
      default_run_env: The environment used when none is requested.
      config_patterns: The per-group glob patterns.
      **kwargs: Additional keyword arguments forwarded to the parent loader.
    """
    # Read from the arguments the loader was actually constructed with, and
    # store them under a private name. The base class does not define `env` or
    # `config_patterns`, so an attribute of that name would resolve to nothing and
    # fail on first access -- which is the only moment anything would notice.
    self._conf_source = conf_source
    self._base_env = base_env
    self._default_run_env = default_run_env
    self._config_patterns = config_patterns
    super().__init__(
      *args,
      conf_source=conf_source,
      base_env=base_env,
      default_run_env=default_run_env,
      config_patterns=config_patterns,
      **kwargs,
    )
    _register_runtime_resolver()
    LOGGER.info(
      'Runtime resolver config loader ready',
      extra={'conf_source': conf_source, 'base_env': base_env, 'default_run_env': default_run_env},
    )


def _register_runtime_resolver() -> None:
  """Register the ``runtime`` OmegaConf resolver exactly once.

  Returns:
    Nothing. Registration is idempotent, so calling this more than once is safe.
  """
  if OmegaConf.has_resolver('runtime'):
    return

  def _resolve(name: str) -> Any:
    """Resolve a name against the current process's runtime parameters.

    The lookup consults, in order, the process environment and the runtime
    parameter store the bootstrap publishes. The environment variable alone was
    the whole implementation and nothing in the framework ever *wrote* it, so
    every ``${runtime:NAME}`` reference resolved to the empty string -- silently,
    because an empty string is a legal value for every configuration type. A
    reference to the run date therefore produced ``''``, which then failed much
    later as a malformed partition predicate.

    A parameter that is genuinely absent still resolves to ``''`` rather than
    raising. That is deliberate: the resolver runs at configuration-load time,
    before the operator can see the run's own parameters in the log, so raising
    there would fail a run over an *optional* reference. An absent key is
    therefore reported at debug level, which is the only channel that can
    distinguish "unset" from "set to empty" without breaking the load.

    Args:
      name: The runtime parameter name.

    Returns:
      The parameter value, or an empty string when it is not set.
    """
    import os  # noqa: PLC0415

    from forecasting_ml_framework.platform.bootstrap import runtime_parameter  # noqa: PLC0415

    value = os.environ.get(f'{ENV_PREFIX}{name}')
    if value is not None:
      return value
    value = runtime_parameter(name)
    if value is None:
      LOGGER.debug(
        'A ${runtime:...} reference resolved to an empty string because the parameter is unset',
        extra={'parameter': name},
      )
      return ''
    return value

  OmegaConf.register_new_resolver('runtime', _resolve, replace=True)
