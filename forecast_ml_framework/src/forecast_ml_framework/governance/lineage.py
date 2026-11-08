#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=import-outside-toplevel

'''Run lineage: inputs, features, model, outputs, config hash.

Pure functions only: no persistence. The caller (or the Kedro
catalog) saves the returned dict. ``config_hash`` is the sha256
of the canonical JSON (sorted keys, compact separators) of the
run's ``variables`` and ``parameters`` config sections.
'''

import copy
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from forecast_ml_framework.core.params import (
  get_first,
)

if TYPE_CHECKING:
  from kedro.io import DataCatalog


__all__ = [
  'COLUMN_LINEAGE_FIELDS',
  'build_lineage',
  'catalog_banner_map',
  'compute_config_hash',
  'get_column_lineage',
  'load_catalog',
  'record_column_lineage',
  'validate_catalog_pii',
]

COLUMN_LINEAGE_FIELDS = (
  'source_dataset',
  'source_column',
  'target_dataset',
  'target_column',
  'transform',
  'run_id',
  'config_hash',
)

_COLUMN_REQUIRED = ('source_column', 'target_column', 'run_id')
_COLUMN_KNOWN = frozenset(
  COLUMN_LINEAGE_FIELDS + ('transformation',))


def _get(source, *names, default=None):
  '''Return the first non-None value among ``names``.

  Dict sources delegate to :func:`core.params.get_first`;
  non-dict sources fall back to ``getattr`` (object run_info).
  ``default`` is returned when no name matches: canonical
  ``get_first`` has no default parameter, so this wrapper
  retains lineage's object and default support.
  '''
  if isinstance(source, dict):
    value = get_first(source, *names)
    return default if value is None else value
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

  Raises:
    ValueError: When ``inputs``, ``outputs`` or
      ``features`` is present but not a list.
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
  for name, value in (('inputs', inputs), ('outputs', outputs),
                       ('features', features)):
    if not isinstance(value, list):
      raise ValueError(
        f'lineage field {name!r} must be a list')
  return {
    'inputs': copy.deepcopy(inputs),
    'features': copy.deepcopy(features),
    'model': copy.deepcopy(model),
    'outputs': copy.deepcopy(outputs),
    'config_hash': compute_config_hash(variables, parameters),
  }


def _resolve_catalog_path(base_dir: str) -> Path:
  '''Resolve a base directory to its catalog.yml path.

  Accepts a catalog.yml file path, a conf/base directory
  containing catalog.yml, or a project root containing
  conf/base/catalog.yml. Only the single base file exists
  in P1-E; env overlays are not merged here.
  '''
  candidate = Path(base_dir)
  if candidate.is_file() and candidate.suffix in ('.yml', '.yaml'):
    return candidate
  for path in (candidate / 'catalog.yml',
               candidate / 'conf' / 'base' / 'catalog.yml'):
    if path.is_file():
      return path
  raise FileNotFoundError(
    f'catalog.yml not found under: {base_dir}')


def validate_catalog_pii(catalog_config: dict) -> None:
  '''Validate catalog metadata PII declarations.

  Entries without metadata or without ``pii_classes``
  are unclassified (allowed, blocks P11 promotion).
  Present declarations must be explicit (including
  ``['none']``) and valid against ``PII_CLASSES``.
  Unknown classes, empty declarations, and mixed
  ``none`` raise ``ValueError`` (config error).
  '''
  from forecast_ml_framework.governance.pii import (
    require_pii_declared)
  if not isinstance(catalog_config, dict):
    raise ValueError('catalog config must be a mapping')
  for name, entry in catalog_config.items():
    if not isinstance(entry, dict):
      continue
    metadata = entry.get('metadata')
    if metadata is None:
      continue
    if not isinstance(metadata, dict):
      raise ValueError(f'catalog {name!r}: metadata must be a map')
    if 'pii_classes' not in metadata:
      continue
    declared = metadata['pii_classes']
    if declared is None:
      continue
    require_pii_declared(pii_classes=declared)


def _banner_map_from_text(text, path) -> dict[str, str]:
  '''Return the banner map from already-read catalog text.

  Args:
    text: Full catalog.yml text (read once by the caller).
    path: Catalog path, for error messages only.

  Returns:
    See :func:`catalog_banner_map`.
  '''
  import yaml
  node = yaml.compose(text)
  if node is None or not isinstance(node, yaml.MappingNode):
    raise ValueError(f'catalog {path} is not a mapping')
  seen: dict[str, int] = {}
  ordered: list[tuple[str, int]] = []
  for key_node, _ in node.value:
    key = str(key_node.value)
    line = int(key_node.start_mark.line) + 1
    if key in seen:
      raise ValueError(
        f'duplicate catalog key {key!r}: '
        f'{path}:{seen[key]} and {path}:{line}')
    seen[key] = line
    ordered.append((key, line))
  banners: list[tuple[int, str]] = []
  for number, raw in enumerate(text.splitlines(), start=1):
    stripped = raw.strip()
    if stripped.startswith('# --- ') and stripped.endswith(' ---'):
      section = stripped[len('# --- '):-len(' ---')].strip()
      if section:
        banners.append((number, section))
  result: dict[str, str] = {}
  for key, line in ordered:
    if key.startswith('_'):
      continue
    section = None
    for banner_line, banner_section in banners:
      if banner_line < line:
        section = banner_section
      else:
        break
    if section is None:
      raise ValueError(
        f'catalog entry {key!r} has no banner ({path}:{line})')
    result[key] = section
  return result


def catalog_banner_map(base_dir: str, text: str | None = None,
                       path: Path | None = None) -> dict[str, str]:
  '''Return {catalog_dataset_name: banner_section}.

  Banners are canonical comment lines ``# --- <section> ---``
  in the single catalog.yml; each dataset name (top-level
  key not starting with ``_``, which is an anchor fragment)
  must sit under exactly one banner. Duplicates are detected
  via a yaml.compose walk naming both 1-based key lines.

  Args:
    base_dir: Base directory, catalog.yml file path, or
      project root (see :func:`_resolve_catalog_path`).
    text: Optional already-read catalog text. Callers
      that already read the file (``load_catalog``) pass
      it here so the file is read only once; when None
      the file is read here.
    path: Optional resolved catalog path for error
      messages; resolved from ``base_dir`` when None.

  Returns:
    Dict mapping each dataset name to its banner section.
  '''
  if text is None:
    path = _resolve_catalog_path(base_dir)
    text = path.read_text(encoding='utf-8')
  elif path is None:
    path = _resolve_catalog_path(base_dir)
  return _banner_map_from_text(text, path)


def load_catalog(base_dir: str) -> 'DataCatalog':
  '''Load the single base catalog as a Kedro DataCatalog.

  Reads ONE catalog.yml exactly once (no fragment-merge
  mechanism in P1-E): the parsed mapping and the banner
  map share the single read. Validates PII metadata
  against PII_CLASSES, enforces banner coverage and
  duplicate-free keys, then returns
  ``DataCatalog.from_config`` on the parsed mapping.
  '''
  import yaml
  path = _resolve_catalog_path(base_dir)
  text = path.read_text(encoding='utf-8')
  config = yaml.safe_load(text)
  if not isinstance(config, dict):
    raise ValueError(f'catalog {path} is not a mapping')
  validate_catalog_pii(config)
  catalog_banner_map(base_dir, text=text, path=path)
  from kedro.io import DataCatalog
  return DataCatalog.from_config(config)


def _is_column_record(value) -> bool:
  '''Return True when ``value`` looks like one record.'''
  if not isinstance(value, dict):
    return False
  return any(
    key in value for key in ('target_column', 'target', 'column'))


def _store_records(store) -> list:
  '''Flatten a caller-supplied store to a record list.

  Args:
    store: List of records, a dict whose values are
      records or (nested) lists/dicts of records, a
      single record dict, or None (no records).

  Returns:
    Records in store order (references, not copies).

  Raises:
    TypeError: If ``store`` is none of the above.
  '''
  if store is None:
    return []
  if isinstance(store, (list, tuple)):
    return [item for item in store if isinstance(item, dict)]
  if isinstance(store, dict):
    if _is_column_record(store):
      return [store]
    records = []
    for value in store.values():
      if isinstance(value, (dict, list, tuple)):
        records.extend(_store_records(value))
    return records
  raise TypeError('store must be a list or dict')


def _validate_column_record(merged: dict) -> dict:
  '''Validate merged fields and return the canonical dict.

  Args:
    merged: Field mapping built from a record and/or
      keyword overrides, with the ``transformation``
      alias already folded into ``transform``.

  Returns:
    Deep-copied dict with exactly
    ``COLUMN_LINEAGE_FIELDS`` keys, in that order.

  Raises:
    ValueError: If a required field is missing or
      blank, or an optional field is neither None
      nor a string.
  '''
  for name in _COLUMN_REQUIRED:
    value = merged.get(name)
    if not isinstance(value, str) or not value.strip():
      raise ValueError(
        f'column lineage field {name!r} must be a '
        'non-empty string')
  for name in COLUMN_LINEAGE_FIELDS:
    if name in _COLUMN_REQUIRED:
      continue
    value = merged.get(name)
    if value is not None and not isinstance(value, str):
      raise ValueError(
        f'column lineage field {name!r} must be a string '
        'or None')
  return {
    name: copy.deepcopy(merged.get(name))
    for name in COLUMN_LINEAGE_FIELDS
  }


def record_column_lineage(record=None, *, source_dataset=None,
                          source_column=None, target_dataset=None,
                          target_column=None, transform=None,
                          transformation=None, run_id=None,
                          config_hash=None) -> dict:
  '''Build a canonical column lineage record.

  Pure function: no persistence. The caller appends
  the returned dict to its own store (list or dict),
  exactly as with :func:`build_lineage`.

  Args:
    record: Optional mapping with any of
      ``COLUMN_LINEAGE_FIELDS`` (``transformation`` is
      accepted as an alias for ``transform``). Unknown
      keys raise ``ValueError``.
    source_dataset: Keyword override; see ``record``.
    source_column: Keyword override; required.
    target_dataset: Keyword override; see ``record``.
    target_column: Keyword override; required.
    transform: Keyword override; see ``record``.
    transformation: Alias for ``transform`` when
      ``transform`` is not supplied.
    run_id: Keyword override; required.
    config_hash: Keyword override; see ``record``.

  Returns:
    Dict with exactly ``COLUMN_LINEAGE_FIELDS`` keys.
    In the legacy keyword form (``source_column``,
    ``target_column``, ``transformation``, ``run_id``
    only) omitted datasets and ``config_hash`` are
    ``None``.

  Raises:
    TypeError: If ``record`` is not a mapping.
    ValueError: If required fields are missing or
      blank, or unknown fields are present.
  '''
  if record is not None and not isinstance(record, dict):
    raise TypeError('record must be a mapping')
  merged = dict(record) if record else {}
  unknown = set(merged) - _COLUMN_KNOWN
  if unknown:
    raise ValueError(
      'unknown column lineage fields: '
      f'{sorted(unknown)}')
  if transform is None:
    transform = transformation
  overrides = {
    'source_dataset': source_dataset,
    'source_column': source_column,
    'target_dataset': target_dataset,
    'target_column': target_column,
    'transform': transform,
    'run_id': run_id,
    'config_hash': config_hash,
  }
  for name, value in overrides.items():
    if value is not None:
      merged[name] = value
  if 'transformation' in merged:
    merged.setdefault('transform', merged.pop('transformation'))
  return _validate_column_record(merged)


def get_column_lineage(dataset=None, column=None,
                       store=None) -> list[dict]:
  '''Query recorded column lineage for a dataset/column.

  Pure function: the store is caller-supplied and is
  never mutated; matches are deep-copied so callers
  cannot corrupt the store through the result.

  Args:
    dataset: Target dataset to match, or None to
      match any dataset. Legacy form: a single string
      positional argument (with ``store=``) is taken
      as ``column`` and matches any dataset.
    column: Target column to match, or None to match
      any column. A list/dict here is taken as
      ``store`` (legacy two-argument form).
    store: Caller-supplied records: a list of record
      dicts, a dict whose values are records or
      lists of records, a single record dict, or
      None. Alternate positional order
      ``(store, dataset, column)`` is also accepted.

  Returns:
    List of deep-copied matching records in store
    order. Unknown dataset/column, an empty store,
    or no store returns ``[]`` (empty list).

  Raises:
    TypeError: If ``store`` is not a list or dict.
  '''
  if (isinstance(dataset, (list, tuple, dict))
      and isinstance(column, str) and isinstance(store, str)):
    dataset, column, store = column, store, dataset
  if isinstance(column, (list, tuple, dict)) and store is None:
    column, store = dataset, column
    dataset = None
  elif column is None and store is not None:
    column, dataset = dataset, None
  matches = []
  for record in _store_records(store):
    target_dataset = _get(
      record, 'target_dataset', 'dataset', default=None)
    target_column = _get(
      record, 'target_column', 'target', 'column', default=None)
    if dataset is not None and target_dataset != dataset:
      continue
    if column is not None and target_column != column:
      continue
    matches.append(copy.deepcopy(record))
  return matches
