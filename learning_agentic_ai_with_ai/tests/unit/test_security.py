#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: security primitives (redaction, sanitization, schema checks).'''


from __future__ import annotations

import pytest

from agentic_common.security import (
  redact_secrets,
  sanitize_untrusted,
  truncate_json,
  validate_against_json_schema,
)


class TestRedactSecrets:
  '''Known secrets never survive redaction.'''

  def test_redacts_env_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GROQ_API_KEYS', 'super-secret-value-123')
    text = 'key is super-secret-value-123 ok'
    redacted = redact_secrets(text)
    assert 'super-secret-value-123' not in redacted
    assert 'REDACTED' in redacted

  def test_extra_secrets(self) -> None:
    text = 'token abcdef1234567890 in text'
    redacted = redact_secrets(text, extra_secrets=['abcdef1234567890'])
    assert 'abcdef1234567890' not in redacted

  def test_clean_text_untouched(self) -> None:
    text = 'nothing to see here'
    assert redact_secrets(text) == text


class TestSanitizeUntrusted:
  '''Untrusted text is cleaned and length-capped.'''

  def test_removes_control_chars(self) -> None:
    dirty = 'a\x00b\x1fc\x7fd'
    cleaned = sanitize_untrusted(dirty)
    for ch in ('\x00', '\x1f', '\x7f'):
      assert ch not in cleaned

  def test_caps_length(self) -> None:
    out = sanitize_untrusted('x' * 10_000, max_chars=100)
    assert len(out) <= 100

  def test_newlines_flattened(self) -> None:
    assert '\n' not in sanitize_untrusted('line1\nline2')


class TestTruncateJson:
  '''JSON payloads are size-capped with a truncation flag.'''

  def test_small_payload_not_truncated(self) -> None:
    result = truncate_json({'a': 1}, max_chars=100)
    assert result['truncated'] is False

  def test_large_payload_truncated(self) -> None:
    result = truncate_json({'blob': 'y' * 10_000}, max_chars=100)
    assert result['truncated'] is True
    assert len(result['data']) <= 100


class TestSchemaValidation:
  '''The minimal JSON-Schema validator enforces the schema subset.'''

  SCHEMA = {
    'type': 'object',
    'required': ['site_id'],
    'properties': {
      'site_id': {'type': 'string'},
      'quantity': {'type': 'integer', 'minimum': 1, 'maximum': 500},
      'priority': {'type': 'string', 'enum': ['low', 'medium', 'high']},
    },
    'additionalProperties': False,
  }

  def test_valid_args_pass(self) -> None:
    errors = validate_against_json_schema(
      {'site_id': 'CS-77', 'quantity': 3, 'priority': 'high'},
      self.SCHEMA,
    )
    assert not errors

  def test_missing_required(self) -> None:
    errors = validate_against_json_schema({'quantity': 1}, self.SCHEMA)
    assert any('site_id' in e for e in errors)

  def test_range_violation(self) -> None:
    errors = validate_against_json_schema(
      {'site_id': 'X', 'quantity': 9000},
      self.SCHEMA,
    )
    assert any('maximum' in e for e in errors)

  def test_enum_violation(self) -> None:
    errors = validate_against_json_schema(
      {'site_id': 'X', 'priority': 'urgent'},
      self.SCHEMA,
    )
    assert any('enum' in e for e in errors)

  def test_type_mismatch(self) -> None:
    errors = validate_against_json_schema({'site_id': 42}, self.SCHEMA)
    assert any('expected type string' in e for e in errors)

  def test_additional_properties_rejected(self) -> None:
    errors = validate_against_json_schema(
      {'site_id': 'X', 'extra': True},
      self.SCHEMA,
    )
    assert any('unexpected property' in e for e in errors)
