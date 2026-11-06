#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Canonical prediction validation shared by serving and scoring.

Single home for the finite check and one-shot iterator
materialization previously duplicated in ``serving/app.py``
and ``pipelines/scoring/nodes.py``. Both call sites import
from here so the traversal semantics cannot drift.
'''

import math


def materialize_predictions(raw):
  '''Materialize a one-shot iterator once into a list.

  Model ``predict`` may return a generator/iterator. The finite
  check and any later conversion would each consume it, so
  checks and conversion must share one materialized list
  instead of exhausting the iterator on the first pass.

  Args:
    raw: Raw model output.

  Returns:
    The same value, or a list when ``raw`` is an iterator.
  '''
  if isinstance(raw, (list, tuple, dict, str, bytes)) or raw is None:
    return raw
  try:
    import numpy as _np
  except ImportError:  # pragma: no cover - numpy always present
    _np = None
  if _np is not None and isinstance(raw, _np.ndarray):
    return raw
  if hasattr(raw, '__next__'):
    return list(raw)
  return raw


def check_predictions_finite(predictions):
  '''Raise ValueError when any numeric prediction is non-finite.

  Traverses dict/list/tuple/ndarray/pandas leaves: bool and
  str labels pass, numeric leaves (float/np/Decimal/complex)
  must be finite, and any other leaf (None, pd.NA, pd.NaT,
  object, bytes) raises. Masked values raise; clean
  MaskedArrays are unwrapped.

  Args:
    predictions: Raw model output (materialize iterators
      first via :func:`materialize_predictions`).

  Raises:
    ValueError: When any leaf is non-finite or of a bad type.
  '''
  from decimal import Decimal
  try:
    import numpy as _np
  except ImportError:  # pragma: no cover - numpy always present
    _np = None
  _masked_array = None
  if _np is not None:
    try:
      _masked_array = _np.ma.MaskedArray
    except Exception:  # noqa: BLE001
      _masked_array = None

  def _unwrap_masked(value):
    '''Raise on masked values; unwrap clean MaskedArrays.'''
    if _masked_array is None or not isinstance(value, _masked_array):
      return value
    if bool(_np.ma.is_masked(value)):
      raise ValueError('predictions must be finite numbers')
    data = getattr(value, 'data', value)
    return _np.asarray(data)

  predictions = _unwrap_masked(predictions)
  if isinstance(predictions, (str, bytes)) or predictions is None:
    stack = [predictions]
  elif isinstance(predictions, dict):
    stack = list(predictions.values())
  elif isinstance(predictions, (list, tuple)):
    stack = list(predictions)
  elif _np is not None and isinstance(
          predictions, _np.ndarray):
    stack = list(predictions.ravel())
  elif isinstance(predictions, (int, float, complex, Decimal)) \
          or (_np is not None and isinstance(
              predictions,
              (_np.floating, _np.integer, _np.complexfloating,
               _np.bool_))):
    stack = [predictions]
  else:
    try:
      converted = predictions.tolist()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
      try:
        stack = list(predictions)
      except TypeError:
        stack = [predictions]
    else:
      stack = list(converted) if isinstance(
          converted, (list, tuple)) else [converted]
  while stack:
    value = stack.pop()
    unwrapped = _unwrap_masked(value)
    if unwrapped is not value:
      stack.append(unwrapped)
      continue
    if isinstance(value, dict):
      stack.extend(value.values())
    elif isinstance(value, (list, tuple)):
      stack.extend(value)
    elif _np is not None and isinstance(
            value, _np.ndarray):
      stack.extend(list(value.ravel()))
    elif hasattr(value, 'tolist') and not isinstance(
            value, (str, bytes)):
      try:
        converted = value.tolist()
      except Exception:  # noqa: BLE001
        raise ValueError(
          'predictions must be finite numbers') from None
      if isinstance(converted, (list, tuple)):
        stack.extend(converted)
      else:
        stack.append(converted)
    elif isinstance(value, bool) or (_np is not None and isinstance(
            value, (_np.bool_,))):
      continue
    elif isinstance(value, str):
      continue
    elif isinstance(value, complex) or (_np is not None and isinstance(
            value, (_np.complexfloating,))):
      try:
        real = float(value.real)
        imag = float(value.imag)
      except (OverflowError, TypeError, ValueError):
        raise ValueError(
          'predictions must be finite numbers') from None
      if not (math.isfinite(real) and math.isfinite(imag)):
        raise ValueError('predictions must be finite numbers')
    elif isinstance(value, (int, float, Decimal)) \
            or (_np is not None and isinstance(
                value, (_np.floating, _np.integer))):
      try:
        number = float(value)
      except OverflowError:
        raise ValueError(
          'predictions must be finite numbers') from None
      except (TypeError, ValueError):
        raise ValueError(
          'predictions must be finite numbers') from None
      if not math.isfinite(number):
        raise ValueError('predictions must be finite numbers')
    else:
      raise ValueError('predictions must be finite numbers')
