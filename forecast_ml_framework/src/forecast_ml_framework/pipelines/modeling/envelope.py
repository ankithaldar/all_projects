#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Single best_model v2 envelope mechanism (Contract 3).'''

import datetime
from typing import Any

import numpy as np

_BASE_KEYS = frozenset({'model', 'label_set', 'registry_ref', 'onnx_sidecar'})
_ERASURE_KEYS = frozenset({'subject_hash', 'requested_at', 'action'})


def unwrap_envelope(obj: Any) -> Any:
  '''Return envelope model if dict envelope, else obj itself.

  This is the single canonical unwrap mechanism for the
  codebase. Callers must import it from this module;
  no local fallback copies are kept elsewhere.

  Args:
    obj: Envelope dict or a bare model object.

  Returns:
    The ``model`` value when ``obj`` is a dict envelope
    holding ``model``, otherwise ``obj`` unchanged.
  '''
  if isinstance(obj, dict) and 'model' in obj:
    return obj['model']
  return obj


def wrap_envelope(model: Any, label_set: list[str] | None,
                  registry_ref: dict | None = None,
                  onnx_sidecar: dict | None = None,
                  task_type: str | None = None) -> dict:
  '''Build v2 dict with exactly the four envelope keys.

  The built envelope is validated via
  :func:`validate_envelope` before it is returned, so an
  invalid envelope (mismatched ``label_set``, missing
  model, malformed sidecar or registry ref) fails loud
  here instead of reaching a caller.

  Args:
    model: Fitted model stored under ``model``.
    label_set: Class labels for classification, or None
      for regression.
    registry_ref: Optional registry reference dict.
    onnx_sidecar: Optional ONNX sidecar dict.
    task_type: Explicit task kind; when None it is
      inferred as classification when ``label_set`` is
      not None or the model exposes ``classes_``, else
      as regression. Any other explicit value raises.

  Returns:
    Dict with exactly ``model``, ``label_set``,
    ``registry_ref`` and ``onnx_sidecar``.

  Raises:
    ValueError: When the built envelope is invalid, per
      :func:`validate_envelope`.
  '''
  env = {
    'model': model,
    'label_set': label_set,
    'registry_ref': registry_ref,
    'onnx_sidecar': onnx_sidecar,
  }
  if task_type is None:
    task = ('classification'
            if label_set is not None
            or model_classes(model) is not None
            else 'regression')
  else:
    task = task_type
  validate_envelope(env, task)
  return env


def model_classes(model: Any) -> np.ndarray | None:
  '''Return classes_ of a fitted object, single-level unwrap.

  Unwraps, in order, at most one envelope level, at most
  one adapter ``model`` attribute, and at most one
  ``_BetaCalWrapper`` ``base`` attribute. This is
  single-level only by design: a doubly wrapped model
  (e.g. a wrapper whose ``base`` is another wrapper)
  is not unwrapped a second time, so its ``classes_``
  lookup returns None instead of the inner base.

  Args:
    model: Envelope, adapter, wrapper, or bare estimator.

  Returns:
    The ``classes_`` array, or None when absent after
    the single-level unwrap.
  '''
  m = unwrap_envelope(model)
  if hasattr(m, 'model'):
    m = m.model
  if type(m).__name__ == '_BetaCalWrapper':
    m = m.base
  return getattr(m, 'classes_', None)


def _is_64_hex(v) -> bool:
  '''Return True when ``v`` is a 64-character hex string.

  Args:
    v: Value to check.

  Returns:
    True for 64-char hex strings, else False.
  '''
  return (
    isinstance(v, str)
    and len(v) == 64
    and all(c in '0123456789abcdefABCDEF' for c in v)
  )


def _is_iso8601(v) -> bool:
  '''Return True when ``v`` is an ISO-8601 string.

  Args:
    v: Value to check.

  Returns:
    True for parseable ISO-8601 strings, else False.
  '''
  if not isinstance(v, str) or not v.strip():
    return False
  text = v.strip()
  try:
    # datetime.fromisoformat in 3.11+ handles Z.
    if text.endswith('Z'):
      datetime.datetime.fromisoformat(text[:-1] + '+00:00')
    else:
      datetime.datetime.fromisoformat(text)
    return True
  except ValueError:
    return False


def validate_envelope(env: dict, task_type: str) -> None:
  '''Assert envelope shape and label_set consistency.

  Checks the exact base keys plus optional ``erasure``,
  label_set/classes_ consistency with the regression
  bijection, onnx_sidecar shape, and registry_ref shape.
  ``onnx_sidecar`` ``sha256`` must be 64-hex when not
  None. ``registry_ref`` must have exactly the keys
  ``{name, alias, version}`` (no forward-compat
  extras).

  Args:
    env: Envelope dict to validate.
    task_type: Task kind, 'classification' or
      'regression' (case-insensitive). Any other
      value is unknown and raises.

  Returns:
    None.

  Raises:
    ValueError: When the envelope shape, label_set,
      task_type, onnx_sidecar, registry_ref, or
      erasure marker is invalid.
  '''
  if not isinstance(env, dict):
    raise ValueError('best_model envelope must be a dict')
  keys = set(env.keys())
  if not _BASE_KEYS.issubset(keys) or set(env.keys()) - _BASE_KEYS - {
      'erasure'}:
    raise ValueError(
      'best_model envelope requires exactly '
      'model/label_set/registry_ref/onnx_sidecar plus '
      'optional erasure')
  if env.get('model') is None:
    raise ValueError('best_model envelope requires model')
  # erasure optional
  if 'erasure' in env:
    er = env['erasure']
    if not isinstance(er, dict) or set(er.keys()) != set(_ERASURE_KEYS):
      raise ValueError(
        'erasure marker must be '
        '{subject_hash, requested_at, action}')
    if not _is_64_hex(er.get('subject_hash')):
      raise ValueError('erasure subject_hash must be 64-hex')
    if not _is_iso8601(er.get('requested_at')):
      raise ValueError('erasure requested_at must be ISO-8601')
    if er.get('action') != 'retrain_required':
      raise ValueError('erasure action must be retrain_required')
  # task normalization: classification/regression,
  # case-insensitive. Any other task_type (including
  # None or empty) is unknown and fails loud.
  tt = str(task_type).strip().lower() if task_type is not None else ''
  is_regression = tt == 'regression'
  is_classification = tt == 'classification'
  if not is_regression and not is_classification:
    raise ValueError('unknown task_type')
  # Require bijection: regression<->None
  label_set = env.get('label_set')
  if is_regression:
    if label_set is not None:
      raise ValueError('label_set must be None for regression')
  elif is_classification:
    if label_set is None:
      raise ValueError('label_set must be a list for classification')
    if not isinstance(label_set, list) or not label_set or not all(
        isinstance(x, str) for x in label_set):
      raise ValueError('label_set must be a non-empty list of str')
    if len(set(label_set)) != len(label_set):
      raise ValueError('label_set must have unique entries')
    classes = model_classes(env)
    if classes is None:
      raise ValueError('model has no classes_ for classification envelope')
    expected = [str(c) for c in classes]
    if label_set != expected:
      raise ValueError('label_set does not match model classes_')
  # onnx_sidecar shape
  side = env.get('onnx_sidecar')
  if side is not None:
    if not isinstance(side, dict):
      raise ValueError('onnx_sidecar must be a dict or None')
    if set(side.keys()) != {'path', 'opset', 'sha256'}:
      raise ValueError('onnx_sidecar must have exactly path/opset/sha256')
    path = side.get('path')
    if not isinstance(path, str) or not path.endswith('.onnx'):
      raise ValueError('onnx_sidecar path must be a .onnx path')
    opset = side.get('opset')
    if isinstance(opset, bool) or not isinstance(opset, int):
      raise ValueError('onnx_sidecar opset must be an int')
    sha = side.get('sha256')
    if sha is not None and not _is_64_hex(sha):
      raise ValueError('onnx_sidecar sha256 must be 64-hex or None')
  # registry_ref shape: exact keys {name, alias, version}
  # (no forward-compat extras).
  ref = env.get('registry_ref')
  if ref is not None:
    if not isinstance(ref, dict):
      raise ValueError('registry_ref must be a dict or None')
    if set(ref.keys()) != {'name', 'alias', 'version'}:
      raise ValueError(
        'registry_ref must have exactly name/alias/version')
    if not isinstance(ref.get('name'), str) or not isinstance(
        ref.get('alias'), str):
      raise ValueError('registry_ref name and alias must be str')
    ver = ref.get('version')
    if ver is not None and not isinstance(ver, str):
      raise ValueError('registry_ref version must be str or None')
