#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YAML configuration loading and class-path instantiation utilities.

This module is the single entry point for turning declarative configuration
into live Python objects. It implements two conventions:

1. Dot-path overrides applied after loading, e.g. ``model.init_params.dropout=0.2``.
2. Recursive instantiation of ``class_path`` / ``init_params`` specifications,
   including lists of such specifications.

Example:
    cfg = load_config('configs/model.yaml')
    model = instantiate(cfg['model'])
"""

from __future__ import annotations

import ast
import importlib
import os
from typing import Any

import yaml


def _parse_scalar(value: str) -> Any:
    """Convert an override string into a Python scalar when possible.

    Args:
        value: Raw string captured from the command line.

    Returns:
        Parsed Python object (bool, int, float, list, dict) or the original string.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.strip().lower()
        mapping = {'true': True, 'false': False, 'null': None, 'none': None}
        return mapping.get(lowered, value)


def _set_by_dot_path(container: dict, dot_path: str, value: Any) -> None:
    """Set a nested key inside a dictionary using a dotted path.

    Args:
        container: Dictionary mutated in place.
        dot_path: Dotted key path, e.g. ``model.init_params.dropout``.
        value: Value to assign at the target location.

    Raises:
        KeyError: If an intermediate path element resolves outside dictionaries.
    """
    keys = dot_path.split('.')
    node = container
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f'Cannot resolve override path at segment: {key!r}')
        node = node[key]
    if not isinstance(node, dict):
        raise KeyError(f'Override target is not a mapping: {dot_path}')
    node[keys[-1]] = value


def _resolve_interpolations(node: Any, root: dict) -> Any:
    """Recursively resolve ``${a.b.c}`` references against the config root.

    Args:
        node: Current node being visited.
        root: Root configuration dictionary used for lookups.

    Returns:
        Node with every interpolation string replaced by its referenced value.

    Raises:
        KeyError: If an interpolated path cannot be resolved.
    """
    if isinstance(node, dict):
        return {key: _resolve_interpolations(value, root) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_interpolations(item, root) for item in node]
    if isinstance(node, str) and node.startswith('${') and node.endswith('}'):
        cursor = root
        for part in node[2:-1].split('.'):
            cursor = cursor[part]
        return cursor
    return node


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load a YAML configuration and apply optional dot-path overrides.

    Args:
        path: Filesystem path to a YAML configuration file.
        overrides: Sequence of ``dotted.path=value`` strings applied after load.

    Returns:
        Fully resolved configuration dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If an override or interpolation path cannot be resolved.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    for override in overrides or []:
        dot_path, _, raw_value = override.partition('=')
        _set_by_dot_path(config, dot_path, _parse_scalar(raw_value))
    return _resolve_interpolations(config, config)


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge ``patch`` into ``base`` without mutating either input.

    Args:
        base: Lower-priority mapping.
        patch: Higher-priority mapping whose leaves win on conflict.

    Returns:
        New merged dictionary; mappings merge recursively, all else replaced.
    """
    merged: dict = dict(base)
    for key, value in patch.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment(path: str, overrides: list[str] | None = None) -> dict:
    """Load an experiment file and compose it from its default config files.

    Experiment files declare:

    * ``defaults``: ordered list of base-config stems resolved next to the
      experiment's parent directory (deep-merged in order).
    * ``override``: experiment-specific section applied last.

    The result is a single self-contained configuration, which callers should
    dump beside run artifacts for traceback.

    Args:
        path: Filesystem path to the experiment YAML
            (e.g. ``configs/experiments/mvp.yaml``).
        overrides: Extra CLI dot-path overrides applied after composition.

    Returns:
        Fully composed and interpolated configuration dictionary.

    Raises:
        KeyError: If a required section or interpolation path is missing.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        experiment = yaml.safe_load(handle) or {}
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    merged: dict = {}
    for stem in experiment.get('defaults', []):
        with open(os.path.join(base_dir, f'{stem}.yaml'), 'r', encoding='utf-8') as handle:
            merged = deep_merge(merged, yaml.safe_load(handle) or {})
    merged = deep_merge(merged, experiment.get('override', {}))
    for override in overrides or []:
        dot_path, _, raw_value = override.partition('=')
        _set_by_dot_path(merged, dot_path, _parse_scalar(raw_value))
    return _resolve_interpolations(merged, merged)


def dump_config(config: dict, path: str) -> None:
    """Persist a resolved configuration for run traceability.

    Args:
        config: Configuration dictionary to serialize.
        path: Destination YAML path (created along with parent directories).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _import_class(class_path: str) -> type:
    """Import a class from its fully qualified path.

    Args:
        class_path: Dotted path such as ``knee.models.knee_net.KneeNet``.

    Returns:
        The imported class object.

    Raises:
        ImportError: If the module or attribute cannot be resolved.
    """
    module_path, _, class_name = class_path.rpartition('.')
    if not module_path:
        raise ImportError(f'Invalid class_path: {class_path!r}')
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f'{class_name!r} not found in {module_path!r}') from exc


def instantiate(spec: Any) -> Any:
    """Recursively resolve a class_path/init_params specification.

    Args:
        spec: A mapping containing ``class_path`` and optional ``init_params``,
            a list of such mappings, or any plain value returned unchanged.

    Returns:
        Instantiated object(s); plain values pass through untouched.
    """
    if isinstance(spec, list):
        return [instantiate(item) for item in spec]
    if isinstance(spec, dict) and 'class_path' in spec:
        cls = _import_class(spec['class_path'])
        params = {
            key: instantiate(value)
            for key, value in (spec.get('init_params') or {}).items()
        }
        return cls(**params)
    return spec
