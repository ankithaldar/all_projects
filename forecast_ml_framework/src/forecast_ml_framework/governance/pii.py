#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''PII vocabulary, detection, masking, and write guards.

All scrubbing and hashing goes through the ``governance.scrub``
leaf (``scrub_text``, ``REDACTED``, ``subject_hash``); this
module implements no scrubber or hasher of its own. Detection
is best-effort column and value heuristics; masking is
deterministic and fail-closed on undeclared PII.

Accepted risks (documented, not code changes): the
detection heuristics may have false negatives on unusual
PII shapes, so declared-class masking never depends on
secret-only scrubbing; and ``subject_hash`` is unsalted
(accepted risk -- deterministic equality preservation
requires it, so the same clear value always hashes to
the same digest).
'''
# pylint: disable=import-outside-toplevel, broad-exception-caught

from dataclasses import dataclass

import re

import pandas as pd

from forecast_ml_framework.governance.scrub import (
    REDACTED, scrub_text, subject_hash,
)

PII_CLASSES: tuple[str, ...] = (
    'email', 'user_id', 'device_id', 'location', 'financial',
    'health', 'none')

_REDACTED_CLASSES = ('email', 'location', 'financial', 'health')
_HASH_CLASSES = ('user_id', 'device_id')

_COLUMN_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('email', ('email', 'e_mail')),
    ('user_id', ('user', 'phone', 'mobile', 'telephone',
                 'full_name', 'fullname', 'first_name',
                 'last_name', 'given_name', 'family_name',
                 'surname')),
    ('device_id', ('device',)),
    ('location', ('location', 'address', 'gps', 'latitude',
                  'longitude')),
    ('financial', ('financial', 'iban', 'credit', 'salary',
                   'income', 'ssn', 'bank', 'tax')),
    ('health', ('health', 'medical', 'diagnosis', 'patient',
                'blood', 'disease', 'treatment')),
)


@dataclass
class PiiFinding:
  '''One detected PII signal (detail is scrubbed, never raw).'''

  column: str
  pii_class: str
  detail: str


def require_pii_declared(*, pii_classes):
  '''Validate a PII declaration, failing closed.

  Args:
    pii_classes: Declared classes; must be explicit, including
      ``['none']`` for datasets with no PII.

  Raises:
    ValueError: When None, empty, mixed with 'none', or unknown.
  '''
  if pii_classes is None:
    raise ValueError('pii_classes must be declared explicitly')
  declared = list(pii_classes)
  if not declared:
    raise ValueError('pii_classes must be declared explicitly')
  unknown = [name for name in declared if name not in PII_CLASSES]
  if unknown:
    raise ValueError(f'unknown pii_classes: {unknown}')
  if 'none' in declared and len(declared) > 1:
    raise ValueError(
        "'none' must be the sole declaration for no-PII data")


def _finding(column, pii_class):
  '''Build a scrubbed finding for one column/class pair.'''
  detail = scrub_text(f'column: {column}; class: {pii_class}')
  return PiiFinding(column=column, pii_class=pii_class,
                    detail=detail)


def detect_pii(df):
  '''Detect PII columns via name and value heuristics.

  Args:
    df: Source DataFrame.

  Returns:
    List of findings, one per column/class pair, with scrubbed
    details that never contain a raw PII value.
  '''
  findings = []
  seen = set()

  def _add(column, pii_class):
    key = (str(column), pii_class)
    if key not in seen:
      seen.add(key)
      findings.append(_finding(column, pii_class))

  for column in list(df.columns):
    lowered = str(column).lower()
    for pii_class, keywords in _COLUMN_KEYWORDS:
      if any(word in lowered for word in keywords):
        _add(column, pii_class)
    series = df[column]
    for value in series.tolist():
      if value is None:
        continue
      try:
        text = str(value)
      except Exception:
        continue
      if re.search(r'[^@\s]+@[^@\s]+\.[^@\s]+', text):
        _add(column, 'email')
      lowered_value = text.lower()
      if lowered_value.startswith('user_') \
          or lowered_value.startswith('user-'):
        _add(column, 'user_id')
      if lowered_value.startswith('device_') \
          or lowered_value.startswith('device-'):
        _add(column, 'device_id')
      # Phone/name are user identifiers; SSN/IBAN are
      # financial. Value patterns catch neutral columns.
      # Separator-optional: continuous digits must not
      # bypass detection via missing separators.
      if re.search(
          r'(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?'
          r'[\s.\-]?\d{3}[\s.\-]?\d{4}\b', text):
        _add(column, 'user_id')
      if re.search(r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b', text):
        _add(column, 'financial')
      if re.search(
          r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b', text,
          re.IGNORECASE):
        _add(column, 'financial')
      if re.search(
          r'\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}\b',
          text, re.IGNORECASE):
        _add(column, 'financial')
      if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text):
        _add(column, 'user_id')
  return findings


def mask_pii(df, *, pii_classes):
  '''Deterministically mask declared PII; fail closed otherwise.

  Args:
    df: Source DataFrame.
    pii_classes: Declared classes (closed vocabulary).

  Returns:
    A masked copy; email/location/financial/health cells become
    ``REDACTED`` and user/device cells become ``subject_hash``
    digests preserving equality.

  Raises:
    ValueError: On an invalid declaration or any undeclared
      finding (no passthrough, no silent leave-unmasked).
  '''
  require_pii_declared(pii_classes=pii_classes)
  declared = list(pii_classes)
  findings = detect_pii(df)
  undeclared = [
      f for f in findings if f.pii_class not in declared]
  if undeclared:
    names = sorted({f.pii_class for f in undeclared})
    raise ValueError(f'undeclared PII classes: {names}')
  masked = df.copy()
  for finding in findings:
    column = finding.column
    if finding.pii_class in _REDACTED_CLASSES:
      masked[column] = REDACTED
    elif finding.pii_class in _HASH_CLASSES:
      masked[column] = masked[column].apply(_hash_cell)
  return masked


def _hash_cell(value):
  '''Hash one cell, preserving missing values.

  Dispatch-free: missing values are recognised by
  identity (``None`` / ``pd.NA`` / ``pd.NaT``) or the
  float-NaN self-inequality, with no ``pd.isna`` call
  and no per-cell import.
  '''
  if value is None or value is pd.NA or value is pd.NaT:
    return value
  if isinstance(value, float) and value != value:
    return value
  return subject_hash(str(value))


def guard_write(df, *, pii_classes, consent_ledger=None,
                entity_col=None):
  '''Fail-closed guard for feature-store writes.

  Validates the declaration, rejects undeclared PII findings,
  and requires a granted consent record per distinct entity
  in the frame whenever declared PII is present.

  Args:
    df: Frame about to be written.
    pii_classes: Declared classes, or None (allowed only when
      no PII is detected; explicit ['none'] asserts no PII).
    consent_ledger: Caller-supplied consent ledger; required
      for declared PII, allowed without only for explicit
      ``['none']`` or None with no detected PII.
    entity_col: Entity column for consent checks. When
      declared PII is present, the entity column must be
      present explicitly: pass ``entity_col`` for a
      non-canonical name, else the frame must carry the
      canonical ``entity_id`` column. Missing raises.

  Raises:
    ValueError: On an invalid declaration, undeclared PII,
      missing entity column for declared PII, missing
      ledger for declared PII, or missing/refused consent
      (fail-closed, no write by caller).
  '''
  if pii_classes is None:
    findings = detect_pii(df)
    if findings:
      names = sorted({f.pii_class for f in findings})
      raise ValueError(f'undeclared PII classes: {names}')
  else:
    require_pii_declared(pii_classes=pii_classes)
    declared = list(pii_classes)
    findings = detect_pii(df)
    undeclared = [
        f for f in findings if f.pii_class not in declared]
    if undeclared:
      names = sorted({f.pii_class for f in undeclared})
      raise ValueError(f'undeclared PII classes: {names}')
  declared_pii = (
      pii_classes is not None
      and list(pii_classes) != ['none'])
  column = entity_col
  if declared_pii:
    if column is None:
      column = 'entity_id'
    if column not in df.columns:
      raise ValueError(
          'entity_col must be declared explicitly when PII'
          ' is declared: missing entity column')
  if consent_ledger is None:
    if pii_classes is None or list(pii_classes) == ['none']:
      return
    raise ValueError(
        'consent ledger required when PII is declared: failing closed')
  try:
    from forecast_ml_framework.governance.consent import (
        check_consent)
  except ImportError as exc:
    raise ValueError(
        'consent ledger unavailable: failing closed') from exc
  column = column or 'entity_id'
  if column not in df.columns:
    raise ValueError(
        'consent check requires an entity column: failing closed')
  seen = []
  for value in df[column].tolist():
    if value not in seen:
      seen.append(value)
  for entity in seen:
    if not check_consent(consent_ledger, entity):
      raise ValueError(
          'consent not granted for entity: failing closed')
