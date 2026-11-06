#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Run lineage: inputs, features, model, outputs, config hash.

Pure functions only: no persistence. The caller (or the Kedro
catalog) saves the returned dict. ``config_hash`` is the sha256
of the canonical JSON (sorted keys, compact separators) of the
run's ``variables`` and ``parameters`` config sections.
'''

import copy
import hashlib
import json


def _get(source, *names, default=None):
  '''Return the first present, non-None value among ``names``.'''
  if isinstance(source, dict):
    for name in names:
      if source.get(name) is not None:
        return source[name]
    return default
  for name in names:
    value = getattr(source, name, None)
    if value is not None:
      return value
  return default


def _config_sections(run_info, variables, parameters):
  '''Extract (variables, parameters) from run_info or kwargs.'''
  config = _get(run_info, 'config', default={})
  if not isinstance(config, dict):
    config = {}
  if variables is None:
    variables = _get(run_info, 'variables', default=None)
  if parameters is None:
    parameters = _get(run_info, 'parameters', default=None)
  if variables is None:
    variables = config.get('variables')
  if parameters is None:
    parameters = config.get('parameters')
  return variables or {}, parameters or {}


def _catalog_names(catalog_entries, role):
  '''Names in catalog entries explicitly tagged with ``role``.

  Supports a list of entries (or dict values) carrying a
  ``role``/``kind``/``direction`` marker. Untagged entries and
  the plain mapping form yield no names: explicit run_info
  fields are the authoritative source for inputs/outputs.
  '''
  if not catalog_entries:
    return []
  if isinstance(catalog_entries, dict):
    entries = list(catalog_entries.values())
  else:
    entries = list(catalog_entries)
  names = []
  for entry in entries:
    if not isinstance(entry, dict):
      continue
    marker = _get(entry, 'role', 'kind', 'direction', default='')
    if str(marker).lower() == role:
      name = _get(entry, 'name', 'dataset', default=None)
      if name is not None:
        names.append(name)
  return names


def _canonical(obj):
  '''Recursively canonicalize sets/frozensets to sorted lists.

  Sets have hash-seed-dependent iteration order, so dumping
  them via ``default=str`` is nondeterministic. Converting
  every set/frozenset to a sorted list (natural ordering,
  falling back to a repr-based ordering for mixed types)
  makes the JSON -- and therefore the hash -- stable.
  '''
  if isinstance(obj, dict):
    return {key: _canonical(value) for key, value in obj.items()}
  if isinstance(obj, (set, frozenset)):
    items = [_canonical(value) for value in obj]
    try:
      return sorted(items)
    except TypeError:
      return sorted(items, key=lambda v: (type(v).__name__, repr(v)))
  if isinstance(obj, (list, tuple)):
    return [_canonical(value) for value in obj]
  return obj


def compute_config_hash(variables, parameters):
  '''Return sha256 of canonical JSON of variables+parameters.

  Args:
    variables: Mapping of config variables.
    parameters: Mapping of pipeline parameters.

  Returns:
    Hex sha256 digest. Stable across key ordering and
    set/frozenset iteration order.
  '''
  canonical = json.dumps(
    _canonical({'variables': variables or {},
                'parameters': parameters or {}}),
    sort_keys=True, separators=(',', ':'), default=str)
  return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def build_lineage(run_info, catalog_entries=None, variables=None,
                  parameters=None):
  '''Build a lineage dict for a run.

  Args:
    run_info: Mapping (or object) describing the run. Recognised
      fields: ``inputs``, ``features``, ``model`` (or
      ``model_name``/``model_id``), ``outputs``, plus
      ``variables`` and ``parameters`` config sections.
    catalog_entries: Optional catalog mapping/list. Used only
      when run_info omits inputs/outputs and entries carry an
      explicit input/output marker.
    variables: Optional override for the variables section.
    parameters: Optional override for the parameters section.

  Returns:
    Dict with keys ``inputs``, ``features``, ``model``,
    ``outputs`` and ``config_hash``. Explicit run_info values
    are preserved verbatim (deep-copied); missing list fields
    default to ``[]`` and a missing model to ``None``.
  '''
  run_info = run_info or {}
  variables, parameters = _config_sections(
    run_info, variables, parameters)
  inputs = _get(run_info, 'inputs', 'input_datasets', default=None)
  if inputs is None:
    inputs = _catalog_names(catalog_entries, 'input')
  outputs = _get(run_info, 'outputs', 'output_datasets', default=None)
  if outputs is None:
    outputs = _catalog_names(catalog_entries, 'output')
  features = _get(run_info, 'features', 'feature_cols', default=None)
  if features is None:
    features = []
  model = _get(
    run_info, 'model', 'model_name', 'model_id', 'model_uri',
    default=None)
  return {
    'inputs': copy.deepcopy(inputs),
    'features': copy.deepcopy(features),
    'model': copy.deepcopy(model),
    'outputs': copy.deepcopy(outputs),
    'config_hash': compute_config_hash(variables, parameters),
  }
