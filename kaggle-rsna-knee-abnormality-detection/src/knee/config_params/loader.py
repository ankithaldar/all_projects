#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Config loading: YAML -> omegaconf merge -> pydantic validation -> objects.

Pipeline
--------
1. ``load_env()`` hydrates secrets from ``.env`` (local) or Kaggle
   Secrets (kernel) so ``${oc.env:VAR}`` interpolations resolve anywhere.
2. YAML is loaded; a sibling/parent ``configs/base.yaml`` (when present)
   is merged underneath so experiments declare only their deltas, then
   CLI ``overrides`` (dotlist) are applied and the result resolved.
3. The plain dict is validated by
   :class:`knee.config_params.schema.ExperimentConfig`.
4. ``instantiate`` turns any ``ComponentSpec`` into a live object via
   importlib -- the single place where class paths become classes
   (Factory / Service-Locator pattern, kept deliberately tiny).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import OmegaConf

from knee.helpers.env import load_env

from .schema import ComponentSpec, ExperimentConfig

T = TypeVar('T')

__all__ = ['load_config', 'instantiate', 'resolve_target']


def _find_base_config(path: Path) -> Path | None:
  """Locate ``configs/base.yaml`` relative to an experiment YAML.

  Args:
      path: Resolved path of the experiment YAML being loaded.

  Returns:
      Base config path when one exists and differs from ``path``,
      otherwise None.
  """
  candidates = [
    path.parent.parent.parent / 'configs' / 'base.yaml',
    path.parent / 'base.yaml',
  ]
  for candidate in candidates:
    if candidate.exists() and candidate.resolve() != path.resolve():
      return candidate
  return None


def load_config(
  path: str, overrides: list[str] | None = None
) -> ExperimentConfig:
  """Load, merge, resolve and validate an experiment YAML.

  Experiment YAMLs hold only deltas against ``configs/base.yaml``:
  whenever a base file sits next to or above the experiment file it is
  merged underneath, so an experiment only declares what makes it
  different (single source of shared defaults).

  Parameters
  ----------
  path:
      Path to the base/experiment YAML file.
  overrides:
      OmegaConf dotlist entries, e.g. ``["train.epochs=3", "seed=7"]``.
  """
  load_env()  # ensure ${oc.env:...} secrets are available before resolve
  cfg_path = Path(path)
  cfg = OmegaConf.load(cfg_path)
  base_path = _find_base_config(cfg_path)
  if base_path is not None:
    cfg = OmegaConf.merge(OmegaConf.load(base_path), cfg)
  if overrides:
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
  OmegaConf.resolve(cfg)
  return ExperimentConfig.model_validate(
    OmegaConf.to_container(cfg, resolve=True)
  )


def resolve_target(dotted: str) -> type:
  """Import ``module.Class`` from a dotted path and return the class."""
  if '.' not in dotted:
    raise ImportError(f"target '{dotted}' must be a dotted module.Class path")
  module_path, _, attr = dotted.rpartition('.')
  module = importlib.import_module(module_path)
  try:
    return getattr(module, attr)
  except AttributeError as exc:  # clearer error than bare AttributeError
    raise ImportError(
      f"module '{module_path}' has no attribute '{attr}'"
    ) from exc


def instantiate(
  spec: ComponentSpec | dict[str, Any], /, **override_params: Any
) -> Any:
  """Build an object from a ComponentSpec; kwargs win over YAML params."""
  if isinstance(spec, dict):
    spec = ComponentSpec.model_validate(spec)
  params: dict[str, Any] = {**spec.params, **override_params}
  cls = resolve_target(spec.target)
  return cls(**params)
