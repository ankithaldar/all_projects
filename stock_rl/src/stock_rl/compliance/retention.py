#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Audit record retention, per record class, with sources attached.

The commonest error in this area is inventing a number and then citing
the wrong instrument for it. This module therefore carries a source
string on every entry, and the table holds only figures traced to a
primary or near-primary document:

  * **8 years** -- books of account and records, SEBI (Stock Brokers)
    Regulations 2026, Regulation 16, notified 7 January 2026. Current
    law, and the reason the algo trail is held for eight years here: the
    trail is captured inside the books-of-account duty. The five-year
    figure in Regulation 18 of the 1992 Regulations was repealed on the
    same date and is not current.
  * **5 years** -- NSE Detailed Operational Modalities (retail algo,
    July 2025) paragraph 10.3: audit trail data should be available for
    at least five years. Corroborated twice in the same document, where
    the system-auditor checklist requires a backup policy and an
    audit-trail policy each of minimum five years. An exchange floor
    binding trading members and API algo providers.
  * **Nothing else.** In particular there is no 7-year figure. It
    appears in design documents attributed to SEBI and has no source at
    all: not Regulation 16, not the 2012 algo circular, not the NSE
    modalities. It is absent from this module deliberately and a test
    asserts it stays absent, because a plausible wrong constant is worse
    than a missing one.

The SEBI algo circulars specify **no retention period at all**.
CIR/MRD/DP/09/2012 paragraph 8(iv) says only that the broker shall
maintain logs of all trading activities to facilitate an audit trail.
The eight-year figure is a broker-records obligation that necessarily
captures algo logs; it is not an algo-specific rule, and quoting "as per
SEBI algo regulations" for any number here would be a false citation.

One record class has no expiry: originals produced to an enforcement
agency are preserved indefinitely (SEBI circular, 4 August 2005). That
is why :func:`retention_years` and :func:`expiry_date` return None for
it. None is a real answer in this module and is never an error, so the
indefinite class can flow through a purge job without a special case.

The Algo ID section lives here rather than in its own module because an
Algo ID is an audit-trail field: it is what makes a record attributable,
which is the entire reason the record is kept.
'''

from __future__ import annotations

from collections.abc import Container, Mapping
from dataclasses import dataclass
from datetime import date, datetime

__all__ = [
  'AlgoIdError',
  'AuditRecordClass',
  'add_years',
  'describe',
  'expiry_date',
  'is_expired',
  'non_algo_id',
  'purge_candidates',
  'registered_algo_ids',
  'require_registration',
  'retention_classes',
  'retention_table',
  'retention_years',
  'source_of',
  'unregistered_algo_id',
  'validate_algo_id',
]


class AlgoIdError(ValueError):
  '''Raised when an Algo ID fails validation.

  A ValueError subclass because an Algo ID is a string and every failure
  mode is a value failure, but with its own type so a caller can tell a
  malformed Algo ID from an unrelated bad argument.
  '''


#: The Algo ID an unregistered client algo sends. NSE permits an
#: unregistered client algo, below the exchange's current
#: orders-per-second threshold of 10, to send this literal with the 13th
#: NNF digit set to the algo flag. The threshold is exchange-set and has
#: changed, so it is venue configuration and is deliberately not encoded
#: here; only the sentinel is.
unregistered_algo_id = '99999'

#: The Algo ID a NON-algo order sends. NSE/FAOP/69296 requires exactly
#: this when the 13th NNF digit is one of the non-algo flags. It is a
#: sentinel, not a registration, and must never be treated as one.
non_algo_id = '0'


@dataclass(frozen=True, slots=True)
class AuditRecordClass:
  '''One class of retained record and the period it must survive.

  Attributes:
    name: Stable identifier, used as the lookup key in
      :data:`retention_table`.
    years: Retention in whole years, or None for indefinite preservation.
    source: The instrument the figure comes from, kept beside the number
      so a citation cannot drift away from the constant it justifies.
    note: Why this period applies to this record class.
  '''

  name: str
  years: int | None
  source: str
  note: str = ''


#: Every record class this project retains, each with its sourced period.
#: There is deliberately no entry with a seven-year period; see the
#: module docstring.
retention_table: tuple[AuditRecordClass, ...] = (
  AuditRecordClass(
    name='books_of_account',
    years=8,
    source='SEBI (Stock Brokers) Regulations 2026, Reg. 16',
    note=('books of account and records; notified 7 Jan 2026, which '
          'repealed the 5-year Reg. 18 of the 1992 Regulations'),
  ),
  AuditRecordClass(
    name='algo_audit_trail',
    years=8,
    source=('SEBI (Stock Brokers) Regulations 2026, Reg. 16; NSE '
            'Detailed Operational Modalities para 10.3'),
    note=('held for 8 rather than the NSE 5-year floor because Reg. 16 '
          'captures the trail as part of the books of account; 5 years '
          'would satisfy only the exchange rule'),
  ),
  AuditRecordClass(
    name='order_and_trade_log',
    years=8,
    source='SEBI (Stock Brokers) Regulations 2026, Reg. 16',
    note=('CIR/MRD/DP/09/2012 para 8(iv) requires logs of all trading '
          'activities, orders, trades and data points and sets no '
          'period of its own'),
  ),
  AuditRecordClass(
    name='exchange_audit_floor',
    years=5,
    source='NSE Detailed Operational Modalities para 10.3',
    note=('the exchange minimum for API and algo audit trails, kept as '
          'its own class so a caller can assert the floor without '
          'confusing it with a recommendation'),
  ),
  AuditRecordClass(
    name='enforcement_original',
    years=None,
    source='SEBI circular, 4 Aug 2005',
    note=('originals produced to an enforcement agency are preserved '
          'indefinitely, so no expiry is computed for this class'),
  ),
)

#: Record classes with a finite period, for a caller that must delete.
retention_classes: tuple[str, ...] = tuple(
  entry.name for entry in retention_table if entry.years is not None
)

_by_name: dict[str, AuditRecordClass] = {
  entry.name: entry for entry in retention_table
}


def _entry(record_class: str) -> AuditRecordClass:
  '''Look up a retention entry.

  Args:
    record_class: One of the names in :data:`retention_table`.

  Returns:
    The matching entry.

  Raises:
    KeyError: If the record class is unknown. Deliberately not a
      ValueError and deliberately not a default: an unrecognised class
      means the caller holds a name that was never registered, and
      falling back to any period is exactly how records get deleted too
      early.
  '''
  entry = _by_name.get(record_class)
  if entry is None:
    raise KeyError(
      f'unknown record class {record_class!r}; known: {sorted(_by_name)}')
  return entry


def retention_years(record_class: str) -> int | None:
  '''Return the retention period for a record class, in whole years.

  Args:
    record_class: One of the names in :data:`retention_table`.

  Returns:
    Years to retain, or None for the indefinite class.

  Raises:
    KeyError: If the record class is unknown.
  '''
  return _entry(record_class).years


def source_of(record_class: str) -> str:
  '''Return the instrument a record class's retention figure comes from.

  Args:
    record_class: One of the names in :data:`retention_table`.

  Returns:
    The source string.

  Raises:
    KeyError: If the record class is unknown.
  '''
  return _entry(record_class).source


def describe(record_class: str) -> str:
  '''Return a one-line audit description of a retention entry.

  Args:
    record_class: One of the names in :data:`retention_table`.

  Returns:
    A human-readable line naming the period and its source.

  Raises:
    KeyError: If the record class is unknown.
  '''
  entry = _entry(record_class)
  period = 'indefinite' if entry.years is None else f'{entry.years} years'
  return f'{entry.name}: retain {period} ({entry.source})'


def expiry_date(record_class: str,
                created: date | datetime) -> date | None:
  '''Return the date on which a record's retention period ends.

  Years, not months and not days: every source uses years, and a
  365-day approximation would drift a February expiry by a day a year.
  Calendar arithmetic is also what an inspection would reproduce, so the
  anniversary rule is used rather than a duration.

  A record created on 29 February expires on 28 February in a non-leap
  year. 29 February does not exist to be the anniversary, and the
  earlier of the two candidate dates is the one that cannot be argued as
  an early deletion.

  Args:
    record_class: One of the names in :data:`retention_table`.
    created: Record creation date, or a datetime whose date is used.

  Returns:
    The first date on which deletion is permitted, or None for the
    indefinite class, which is a real answer rather than an error.

  Raises:
    KeyError: If the record class is unknown.
  '''
  years = retention_years(record_class)
  if years is None:
    return None
  start = created.date() if isinstance(created, datetime) else created
  return add_years(start, years)


def is_expired(record_class: str, created: date | datetime,
               today: date | datetime) -> bool:
  '''Return True once a record may lawfully be deleted.

  Expiry is inclusive: a record whose period ends on the 1st is expired
  as of the 1st, not the 2nd.

  Args:
    record_class: One of the names in :data:`retention_table`.
    created: Record creation date or datetime.
    today: Date or datetime to test against.

  Returns:
    True if the retention period has run out. Always False for the
    indefinite class, which is the safe direction to be wrong in: a
    caller that skips a deletion has kept a record it was allowed to
    keep.

  Raises:
    KeyError: If the record class is unknown.
  '''
  expiry = expiry_date(record_class, created)
  if expiry is None:
    return False
  current = today.date() if isinstance(today, datetime) else today
  return current >= expiry


def purge_candidates(record_class: str, records: Mapping[str, date],
                     today: date) -> list[str]:
  '''Return the keys of records whose retention period has run out.

  A pure function, and the only helper here that looks at a collection
  of records at all, so that a dry run is the default. A helper that
  computes and deletes in one step cannot be reviewed, and this one is
  the step before a deletion.

  Args:
    record_class: One of the names in :data:`retention_table`.
    records: Mapping of record key to its creation date.
    today: Date to evaluate expiry against.

  Returns:
    Sorted list of expired keys. Empty for the indefinite class.

  Raises:
    KeyError: If the record class is unknown. The class is resolved
      before ``records`` is walked, so a typo fails even on an empty
      mapping: an empty result from a purge dry run is
      indistinguishable from "nothing to delete", which is the one
      reading a caller would act on.
  '''
  _entry(record_class)
  return sorted(
    key for key, created in records.items()
    if is_expired(record_class, created, today)
  )


def add_years(start: date, years: int) -> date:
  '''Return ``start`` shifted forward by whole calendar years.

  Public because a caller holding a custom retention period -- a client
  agreement with its own term, say -- needs the same calendar arithmetic
  the expiry helper uses, and reimplementing it as a 365-day duration is
  exactly the drift this module exists to avoid.

  Args:
    start: Date to shift.
    years: Whole years to add. Must be non-negative; retention is a
      forward period and a caller wanting to go backwards has a bug.

  Returns:
    The anniversary, clamped to 28 February when the source day does not
    exist in the target year.

  Raises:
    ValueError: If ``years`` is negative.
  '''
  if years < 0:
    raise ValueError(f'years must be >= 0, got {years}')
  try:
    return start.replace(year=start.year + years)
  except ValueError:
    # Only 29 February reaches here. 28 February exists in every year,
    # so it cannot be argued as an early deletion.
    return start.replace(year=start.year + years, day=28)


def registered_algo_ids(approved: Container[str]) -> frozenset[str]:
  '''Return the Algo IDs valid on this venue, sentinels included.

  Registration is per venue and per strategy: the same code is a
  different algo to NSE and to BSE, and one algo needs separate
  registration on each exchange. An Algo ID is therefore an opaque,
  exchange-namespaced string and never an integer. NSE's own sentinel for
  an unregistered client algo is the numeric-looking string ``99999``,
  which an integer-typed field would coerce into a different value, and
  the string form is the one the exchange documents.

  Both sentinels are folded in so that the ordinary below-threshold
  client algo and an ordinary non-algo order both validate without a
  special case at every call site. Membership in this set means "will
  the exchange accept it", not "is this algo registered"; use
  :func:`require_registration` for the second question.

  Args:
    approved: Any container of the Algo IDs registered on this venue.

  Returns:
    A frozenset of those IDs plus the two sentinels.
  '''
  return frozenset(approved) | {unregistered_algo_id, non_algo_id}


def validate_algo_id(algo_id: str, registered: Container[str]) -> str:
  '''Validate an Algo ID and return it unchanged.

  The order of the checks is the point, which is why emptiness and
  whitespace are separate tests rather than one set membership test. An
  empty string is the signature of an unpopulated field and whitespace is
  the signature of a value that has been through a spreadsheet. Both
  mean the Algo ID was never really set, and both must fail before
  membership is considered, because a blank is not a registration and
  must never be quietly coerced to the non-algo sentinel.

  Args:
    algo_id: The Algo ID exactly as it will be sent to the exchange.
    registered: The IDs accepted on this venue, typically from
      :func:`registered_algo_ids`.

  Returns:
    The Algo ID, unchanged. Returned rather than discarded so a caller
    can write ``algo_id = validate_algo_id(raw, approved)`` and have
    the audit trail hold the value that was actually checked.

  Raises:
    AlgoIdError: If the ID is empty, whitespace-only, padded, or absent
      from the registered set.
    TypeError: If ``algo_id`` is not a string. An integer here is the
      failure this module exists to prevent, because NSE's sentinel is
      numeric-looking.
  '''
  if not isinstance(algo_id, str):
    raise TypeError(
      f'algo_id must be an opaque string, got {type(algo_id).__name__}')
  if not algo_id:
    raise AlgoIdError('algo_id is empty; an unregistered client algo '
                      f'sends {unregistered_algo_id!r}')
  if not algo_id.strip():
    raise AlgoIdError(
      f'algo_id is whitespace only: {algo_id!r}; an unregistered client '
      f'algo sends {unregistered_algo_id!r}')
  if algo_id != algo_id.strip():
    raise AlgoIdError(f'algo_id has surrounding whitespace: {algo_id!r}')
  if algo_id not in registered:
    raise AlgoIdError(
      f'algo_id {algo_id!r} is not registered on this venue; an algo '
      'without exchange approval must not trade')
  return algo_id


def require_registration(algo_id: str) -> bool:
  '''Return True if an Algo ID needs registered rather than sentinel.

  A client running a sub-threshold unregistered algo legitimately sends
  ``99999``. A desk claiming exchange approval must send a real ID, and
  the difference decides which retention duty applies: an unregistered
  algo carries the exchange floor, a registered one falls inside the
  books-of-account obligation.

  Args:
    algo_id: The Algo ID to classify.

  Returns:
    True for anything that is not one of the two sentinels.
  '''
  return algo_id not in (unregistered_algo_id, non_algo_id)
