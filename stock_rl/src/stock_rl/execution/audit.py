#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Append-only decision log with a hash chain, and no external store.

**What this is for.** NSE operational modalities 9.1 and 9.9: no
modification may be made to a registered black-box algo, and fresh
Exchange registration is required for **any** change to the logic
governing it. This system is a black box. So the question "is the running
process the registered one?" has to be answerable, and the only thing in
this file that answers it is the **policy fingerprint** from
:func:`stock_rl.rl.policy.policy_fingerprint`, recorded on every decision.
Not the feature values, not the rule ids: those describe the decision.
The fingerprint describes the *code and parameters* that made it, which
is exactly what the re-registration obligation is about.

**Honest framing, because the design doc implies otherwise.** No Indian
rule requires a per-decision rationale. Not one. CIR/MRD/DP/09/2012 para
8(iv) requires "logs of all trading activities", orders, trades, data
points and control parameters -- an obligation to *record*, not to
*explain*. The Feb 2025 circular's requirements are per algo and static,
at registration. MIFID II, the FCA's 2025 multi-firm review and IOSCO
PD788 all press for explainability and none of them defines a sufficient
explanation. So this log is an **internal control**, justified as
strategy-identity evidence: the mechanism that lets us demonstrate the
deployed behaviour matches the registered behaviour. Building it is
right. Describing it as a regulatory requirement would be a fabricated
citation, which is the one failure this project is built to avoid.

**Retention.** The floor is 5 years -- NSE Detailed Operational Modalities
para 10.3, "the audit trail data should be available for at least 5
years" -- and the choice here is **8 years**, from SEBI (Stock Brokers)
Regulations 2026 reg. 16 (books of account and records, notified 7 Jan
2026, which repealed the 5-year reg. 18 of the 1992 Regulations). The
eight-year figure is the conservative reading because reg. 16 captures
the trail inside the books-of-account duty. **There is no 7-year figure
in any source**; it appears in design documents attributed to SEBI and
has no instrument behind it. This module takes its period from
:mod:`stock_rl.compliance.retention` rather than hardcoding a number, so
the citation cannot drift away from the constant.

**The hash chain is the tamper evidence.** Each record carries the
previous record's hash, so editing record *n* invalidates record *n*
itself and every link after it. :func:`verify_chain` walks the file and
reports the first sequence number that fails, so a tampering attempt
produces a *position*, not just a boolean. An append-only file with a
plain checksum per line would not do: an attacker who edits a record
simply recomputes that record's checksum, and only a chain resists.

**The chain is not a digital signature and does not pretend to be.** It
detects accidental corruption and it detects an editor who did not know
the chain was there. An attacker with write access to the file can
recompute the whole chain from their own edit. Defending against that
needs an external witness -- a timestamp authority, an append-only
object store, a co-signed record -- and none of those is a standard
library feature. Stating the limit is more useful than implying the
guarantee is stronger than it is.

**Append is durable and fail-loud.** Each record is one JSON line,
written and flushed. A refusal to write raises; it does not return
quietly, because an audit log that can lose records without saying so is
worse than no audit log, because it looks like one.
'''

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from stock_rl.compliance.retention import (
  expiry_date,
  retention_years,
  source_of,
)

__all__ = [
  'AuditChainError',
  'AuditLog',
  'AuditRecord',
  'ChainVerification',
  'audit_schema',
  'decision_log_class',
  'feature_hash',
  'genesis_hash',
  'hash_payload',
]

#: Schema tag written into every record. Bump it when the record shape
#: changes, so an old chain cannot be verified as though it were a new
#: one.
audit_schema = 'stock_rl.execution.audit/1'

#: The retention class this log belongs to, as named in
#: :mod:`stock_rl.compliance.retention`. The period and its source are
#: read from that table rather than written here, so a citation cannot
#: drift away from the number it justifies.
decision_log_class = 'algo_audit_trail'

#: Prev-hash of the first record in a chain.
genesis_hash = '0' * 64

#: Source of wall-clock time as an aware UTC datetime. Injectable so a
#: test produces a byte-identical log and asserts on the file itself.
Clock = Callable[[], datetime]


def utcnow() -> datetime:
  '''Return the current time as an aware UTC datetime.

  Returns:
    Timezone-aware datetime in UTC.
  '''
  return datetime.now(timezone.utc)


def hash_payload(payload: Mapping[str, object]) -> str:
  '''Return the SHA-256 of a canonical JSON rendering of a payload.

  Canonical means ``sort_keys`` with no insignificant whitespace, so the
  same logical record hashes identically regardless of the order its
  fields were built in. Without that, two processes writing the same
  record would produce two hashes and the chain would break on
  replication for no reason.

  Args:
    payload: JSON-stable mapping. Values must be JSON-serialisable.

  Returns:
    Lowercase hex digest.
  '''
  text = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                    ensure_ascii=True)
  return hashlib.sha256(text.encode('utf-8')).hexdigest()


def feature_hash(features: Mapping[str, float]) -> str:
  '''Return a stable hash of the feature vector at decision time.

  Floats are rendered through ``repr`` so the digest records the exact
  double rather than a rounded decimal, and keys are sorted so the hash
  does not depend on insertion order. Two states that differ in the
  fifteenth decimal place produce different hashes, which is the point:
  the hash answers "was this the same state", not "was this the same to
  six figures".

  Args:
    features: Feature name to value. Values must be finite floats.

  Returns:
    Lowercase hex digest of the canonical rendering.

  Raises:
    ValueError: If any value is not finite. A NaN or an infinity would
      hash consistently to the same string regardless of what it means,
      so two genuinely different states could share a digest.
  '''
  payload: dict[str, object] = {}
  for key in sorted(features):
    value = float(features[key])
    if value != value or value in (float('inf'), float('-inf')):
      raise ValueError(
        f'feature {key!r} is not finite; a non-finite value would hash to '
        'the same digest whatever it meant')
    payload[str(key)] = repr(value)
  return hash_payload(payload)


@dataclass(frozen=True, slots=True)
class AuditRecord:
  '''One decision, as recorded.

  Attributes:
    seq: One-based position in the chain.
    at: Decision time, ISO 8601 UTC.
    decision: What was decided, e.g. ``'buy'``.
    rule_ids: Identifiers of the rules or conditions that fired. Free
      text is permitted because the rule tree is the caller's, but the
      identifiers are what makes a decision greppable later.
    indicators: Indicator values **at decision time**, as recorded. Not
      recomputed and not back-filled: the value the rule actually saw is
      the value that has to be on the record.
    timeframe: Timeframe the decision was taken on.
    timeframe_criterion: Why that timeframe was chosen. Per-decision
      timeframe rationale is our own engineering choice, same standing as
      the rest of this record.
    policy_fingerprint: From
      :func:`stock_rl.rl.policy.policy_fingerprint`. The field that
      discharges the 9.1 / 9.9 re-registration obligation.
    features: Hash of the feature vector, see :func:`feature_hash`.
    prev_hash: The previous record's ``record_hash``, or
      :data:`genesis_hash` for the first.
    record_hash: SHA-256 over every other field.
  '''

  seq: int
  at: str
  decision: str
  rule_ids: tuple[str, ...]
  indicators: Mapping[str, float]
  timeframe: str
  timeframe_criterion: str
  policy_fingerprint: str
  features: str
  prev_hash: str
  record_hash: str = ''

  def payload(self) -> dict[str, object]:
    '''Return the canonical payload the record hash covers.

    ``record_hash`` itself is excluded, obviously, since it is the hash
    of everything else.

    Returns:
      Mapping of every field except ``record_hash``.
    '''
    return {
      'schema': audit_schema,
      'seq': self.seq,
      'at': self.at,
      'decision': self.decision,
      'rule_ids': list(self.rule_ids),
      'indicators': {key: repr(float(value))
                     for key, value in sorted(self.indicators.items())},
      'timeframe': self.timeframe,
      'timeframe_criterion': self.timeframe_criterion,
      'policy_fingerprint': self.policy_fingerprint,
      'features': self.features,
      'prev_hash': self.prev_hash,
    }

  def sealed(self) -> 'AuditRecord':
    '''Return a copy carrying its own hash.

    The hash covers every field except itself, via :meth:`payload`, so
    sealing is idempotent: re-sealing a sealed record reproduces the same
    digest, which is what lets a verifier recompute it without knowing
    whether it was sealed before being written.

    Returns:
      The record with ``record_hash`` filled in.
    '''
    digest = hash_payload(self.payload())
    values = {item.name: getattr(self, item.name) for item in fields(self)}
    values['record_hash'] = digest
    return AuditRecord(**values)

  def to_json(self) -> dict[str, object]:
    '''Return the record as JSON-stable primitives.

    Returns:
      Mapping with string keys and JSON-safe values, ready for one line
      of the log.
    '''
    return {
      'schema': audit_schema,
      'seq': self.seq,
      'at': self.at,
      'decision': self.decision,
      'rule_ids': list(self.rule_ids),
      'indicators': {key: repr(float(value))
                     for key, value in sorted(self.indicators.items())},
      'timeframe': self.timeframe,
      'timeframe_criterion': self.timeframe_criterion,
      'policy_fingerprint': self.policy_fingerprint,
      'features': self.features,
      'prev_hash': self.prev_hash,
      'record_hash': self.record_hash,
    }

  @classmethod
  def from_json(cls, payload: Mapping[str, object]) -> 'AuditRecord':
    '''Rebuild a record from one parsed line of the log.

    Floats are recovered through ``float()`` from their ``repr`` text,
    which round-trips exactly for a Python double, so the indicator
    values a record reports are the values that were recorded rather
    than the nearest decimal to them.

    Args:
      payload: Mapping produced by :meth:`to_json`.

    Returns:
      The :class:`AuditRecord`.

    Raises:
      ValueError: If a required key is missing, the schema tag is
        unknown, or a value will not parse. A record that cannot be read
        back cannot be verified, so a silent default here would break
        the chain without saying so.
    '''
    schema = payload.get('schema')
    if schema != audit_schema:
      raise ValueError(
        f'record schema {schema!r} is not {audit_schema!r}')
    try:
      indicators = {
        str(key): float(str(value))
        for key, value in dict(payload['indicators']).items()  # type: ignore[arg-type]
      }
      return cls(
        seq=int(payload['seq']),  # type: ignore[arg-type]
        at=str(payload['at']),
        decision=str(payload['decision']),
        rule_ids=tuple(str(item) for item in payload['rule_ids']),  # type: ignore[union-attr]
        indicators=indicators,
        timeframe=str(payload['timeframe']),
        timeframe_criterion=str(payload['timeframe_criterion']),
        policy_fingerprint=str(payload['policy_fingerprint']),
        features=str(payload['features']),
        prev_hash=str(payload['prev_hash']),
        record_hash=str(payload['record_hash']),
      )
    except (KeyError, TypeError, ValueError) as exc:
      raise ValueError(f'malformed audit record: {exc}') from exc


@dataclass(frozen=True, slots=True)
class ChainVerification:
  '''The result of walking a chain.

  Attributes:
    ok: Whether every record verified and every link matched.
    checked: Number of records that **verified**. On a failure this is
      the count of good records *before* the break, not the size of the
      file, because a file whose length an editor chose is not evidence.
    first_bad_seq: Sequence number of the first record that failed, or
      None when the chain is intact. A *position*, not a boolean, because
      "the log was edited" is much less useful to an operator than
      "record 41 was edited".
    reason: Why verification failed, or a statement that it passed.
  '''

  ok: bool
  checked: int
  first_bad_seq: int | None = None
  reason: str = ''

  @property
  def verified(self) -> bool:
    '''Return whether the chain is intact. Alias of :attr:`ok`.'''
    return self.ok

  def render(self) -> str:
    '''Return a one-line summary for a log.

    Returns:
      Human-readable summary naming the first bad record when there is
      one, and the number of records verified ahead of it.
    '''
    if self.ok:
      return f'chain verified over {self.checked} records'
    return (
      f'CHAIN BROKEN at record {self.first_bad_seq} after {self.checked} '
      f'verified: {self.reason}')


class AuditChainError(RuntimeError):
  '''Raised when a chain is malformed and the log refuses to extend it.

  Distinct from :class:`ValueError` because this is not a bad argument:
  the file is bad, and the caller cannot fix it by passing something
  else. It carries the verification so the caller can log *where*.
  '''


class AuditLog:
  '''An append-only JSONL decision log with a hash chain.

  Args are documented on ``__init__``.
  '''

  path: Path
  policy_fingerprint: str
  record_class: str
  retention: int | None

  def __init__(
    self,
    path: str | Path,
    policy_fingerprint: str,
    clock: Clock = utcnow,
    record_class: str = decision_log_class,
  ) -> None:
    '''Open or create a decision log.

    Existing state is read and verified on construction, so a new process
    appends to a chain it has confirmed rather than to whatever is on
    disk. A broken chain does **not** raise here -- a log must stay
    readable when it is broken, or the tampering cannot be inspected --
    but :meth:`append` refuses on it, which is the part that matters.

    Args:
      path: Path to the JSONL file. Parent directories are created on
        first append.
      policy_fingerprint: From
        :func:`stock_rl.rl.policy.policy_fingerprint`. Required and
        non-blank: a decision recorded without the identity of the code
        that made it does not discharge 9.1 / 9.9, so it is refused
        rather than stored as empty.
      clock: Wall-clock source, injectable for byte-identical tests.
      record_class: Retention class name, resolved through
        :mod:`stock_rl.compliance.retention`.

    Raises:
      ValueError: If the fingerprint is blank.
      KeyError: If the retention class is not one of
        :data:`stock_rl.compliance.retention.retention_table`. A KeyError
        and not a default, because an unknown class must not silently
        inherit another record class's retention period.
    '''
    if not isinstance(policy_fingerprint, str) or \
        not policy_fingerprint.strip():
      raise ValueError(
        'policy_fingerprint must be a non-blank string; a decision '
        'recorded without the identity of the code that made it does not '
        'discharge the NSE 9.1/9.9 re-registration obligation')
    self.path = Path(path)
    self.policy_fingerprint = policy_fingerprint
    self.record_class = record_class
    self._retention = retention_years(record_class)
    self._clock = clock
    self._verification = self._walk()

  @property
  def retention_years(self) -> int | None:
    '''Return the retention period for this record class, in years.

    Sourced from :mod:`stock_rl.compliance.retention` rather than
    hardcoded, so the eight-year choice and its citation cannot drift
    apart. None means indefinite preservation, which is a real answer
    and not an error.
    '''
    return self._retention

  @property
  def retention_source(self) -> str:
    '''Return the instrument the retention figure comes from.'''
    return source_of(self.record_class)

  def expires_on(self, created: datetime) -> datetime | None:
    '''Return when a record created at ``created`` may be purged.

    Args:
      created: Creation timestamp of a record.

    Returns:
      The first timestamp at which deletion is permitted, or None for an
      indefinite class. Expiry is computed with calendar arithmetic in
      :mod:`stock_rl.compliance.retention`, not with a 365-day
      approximation, because an approximation drifts a February expiry by
      a day every year.
    '''
    date = expiry_date(self.record_class, created)
    return None if date is None else datetime(date.year, date.month,
                                             date.day)

  def append(
    self,
    decision: str,
    rule_ids: tuple[str, ...] | list[str] = (),
    indicators: Mapping[str, float] | None = None,
    timeframe: str = 'daily',
    timeframe_criterion: str = '',
    features: Mapping[str, float] | None = None,
    feature_digest: str = '',
    at: datetime | None = None,
  ) -> AuditRecord:
    '''Append one decision to the chain.

    The record is hashed, written as one line, flushed, and the in-memory
    chain advanced. If the write fails the exception propagates: an audit
    log that loses records silently is worse than none, because it is
    believed.

    Args:
      decision: What was decided. Must be non-blank.
      rule_ids: Identifiers of the rules or conditions that fired.
      indicators: Indicator values as the rules saw them.
      timeframe: Timeframe the decision was taken on.
      timeframe_criterion: Why that timeframe, recorded because a
        per-decision timeframe rationale is this project's own control
        and is worth having in the record rather than in a comment.
      features: Feature vector, hashed by :func:`feature_hash`.
      feature_digest: A pre-computed feature hash, for a caller holding
        a vector too large to re-render. Mutually exclusive with
        ``features``; supplying both is refused rather than silently
        preferring one.
      at: Decision time, defaulting to the injected clock.

    Returns:
      The sealed :class:`AuditRecord` as written.

    Raises:
      ValueError: If the decision is blank or both feature forms are
        given.
      AuditChainError: If the existing chain on disk does not verify. A
        log with a broken chain is evidence; appending to it would bury
        the evidence under more records.
      OSError: If the record cannot be written.
    '''
    if not isinstance(decision, str) or not decision.strip():
      raise ValueError(
        'decision must be a non-blank string; an unattributed decision '
        'is not a record')
    if features is not None and feature_digest:
      raise ValueError(
        'supply either features or feature_digest, not both: a record '
        'carrying two different notions of its own state is worse than '
        'one carrying neither')
    if not self._verification.ok:
      raise AuditChainError(
        f'refusing to extend a broken chain: {self._verification.render()}')
    values = dict(indicators or {})
    digest = feature_digest or feature_hash(features or {})
    last_hash = self._last_hash()
    record = AuditRecord(
      seq=self._verification.checked + 1,
      at=_stamp(at if at is not None else self._clock()),
      decision=decision,
      rule_ids=tuple(rule_ids),
      indicators=values,
      timeframe=timeframe,
      timeframe_criterion=timeframe_criterion,
      policy_fingerprint=self.policy_fingerprint,
      features=digest,
      prev_hash=last_hash,
    ).sealed()
    line = json.dumps(record.to_json(), sort_keys=True,
                      separators=(',', ':'), ensure_ascii=True)
    try:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      with self.path.open('a', encoding='utf-8') as handle:
        handle.write(f'{line}\n')
        handle.flush()
    except OSError as exc:
      raise AuditChainError(
        f'audit record for {record.decision!r} could not be written to '
        f'{self.path} ({exc}); a decision that is not recorded is a '
        'decision nobody can account for') from exc
    self._verification = ChainVerification(
      ok=True, checked=record.seq, reason='chain extended in memory')
    return record

  def records(self) -> tuple[AuditRecord, ...]:
    '''Return every record in the file, in order.

    Returns:
      The parsed records. A line that will not parse raises, because a
        reader that skips it is a reader that cannot verify the chain.
    '''
    return tuple(_read(self.path))

  def verify_chain(self) -> ChainVerification:
    '''Re-read the file and verify every record and every link.

    Called on demand rather than trusted from construction, so a file
    edited behind this object's back is detected rather than assumed
    unchanged.

    Returns:
      A :class:`ChainVerification` naming the first record that failed.
    '''
    self._verification = self._walk()
    return self._verification

  def _walk(self) -> ChainVerification:
    '''Verify the chain currently on disk.

    Returns:
      The verification result, from :func:`verify_chain`. A missing file
      verifies trivially, which is correct: an empty chain has nothing
      to contradict.
    '''
    return verify_chain(self.path)

  def _last_hash(self) -> str:
    '''Return the hash the next record must chain onto.

    Returns:
      The final record's hash, or :data:`genesis_hash` for an empty log.
      Re-read from disk rather than trusted from memory, so a log
      appended to by another process is chained onto correctly.
    '''
    if not self.path.exists():
      return genesis_hash
    last: AuditRecord | None = None
    for record in _read(self.path):
      last = record
    return genesis_hash if last is None else last.record_hash

  def render(self) -> str:
    '''Return a one-line status for a log.

    Returns:
      Human-readable status naming the record count, the fingerprint and
      the retention period with its source.
    '''
    years = ('indefinite' if self.retention_years is None
             else f'{self.retention_years} years')
    return (
      f'audit log {self.path}: {self._verification.checked} records, '
      f'retain {years} ({self.retention_source})')


def verify_chain(path: str | Path) -> ChainVerification:
  '''Verify a JSONL chain on disk, reporting where it fails.

  Standalone so a log can be inspected -- by an auditor, a CI job or a
  shell one-liner -- without opening it for append. It re-reads the file
  rather than trusting anything cached, which is the only way a file
  edited behind a running process is noticed.

  Three distinct failures are distinguished, because they mean different
  things to whoever has to fix them:

    * **content mismatch**: a record's own hash does not match its
      contents, so that record was edited;
    * **link mismatch**: a record's ``prev_hash`` is not its
      predecessor's hash, so a record was edited, removed or reordered;
    * **sequence mismatch**: the file has a gap or a renumbering, so a
      record was deleted without touching the rest.

  Args:
    path: Path to the JSONL log. A missing file verifies as an empty
      chain.

  Returns:
    A :class:`ChainVerification`. ``first_bad_seq`` is set to the
    offending record's sequence number on failure.
  '''
  target = Path(path)
  if not target.exists():
    return ChainVerification(True, 0, reason='no records yet')
  expected = genesis_hash
  verified = 0
  try:
    for record in _read(target):
      position = verified + 1
      # The link is tested before the sequence number, so a *deleted*
      # record is reported as a broken chain rather than as a
      # renumbering. A removal and an edit are different events and the
      # operator's next step differs.
      if record.prev_hash != expected:
        return ChainVerification(
          False, verified, record.seq,
          f'prev_hash {record.prev_hash[:12]} does not match the previous '
          f'record hash {expected[:12]}: the chain was broken between '
          'records')
      if record.seq != position:
        return ChainVerification(
          False, verified, record.seq,
          f'record found at position {position} claims seq {record.seq}: a '
          'record was added, removed or renumbered')
      digest = hash_payload(record.payload())
      if digest != record.record_hash:
        return ChainVerification(
          False, verified, record.seq,
          f'record hash {record.record_hash[:12]} does not match its '
          f'contents {digest[:12]}: this record was edited')
      verified += 1
      expected = record.record_hash
  except (ValueError, OSError) as exc:
    return ChainVerification(
      False, verified, verified + 1, f'unreadable record: {exc}')
  return ChainVerification(True, verified, reason='chain verified')


def _read(path: Path) -> list[AuditRecord]:
  '''Parse every line of a JSONL log.

  Args:
    path: The log file. Must exist.

  Returns:
    Records in file order. Blank trailing lines are skipped; a line with
    content that will not parse raises.
  '''
  records: list[AuditRecord] = []
  with path.open(encoding='utf-8') as handle:
    for number, line in enumerate(handle, start=1):
      text = line.strip()
      if not text:
        continue
      try:
        payload = json.loads(text)
      except json.JSONDecodeError as exc:
        raise ValueError(f'line {number} is not valid JSON: {exc}') from exc
      if not isinstance(payload, dict):
        raise ValueError(f'line {number} is not a JSON object')
      records.append(AuditRecord.from_json(payload))
  return records


def _stamp(moment: datetime) -> str:
  '''Return an ISO 8601 timestamp.

  Args:
    moment: The time to render. A naive value is stamped as UTC.

  Returns:
    The time in ISO 8601, UTC.
  '''
  if moment.tzinfo is None:
    moment = moment.replace(tzinfo=timezone.utc)
  return moment.isoformat()
