#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Translation of warehouse SQL into the SQLite dialect.

The catalog's queries are written once, for the production warehouse. A local
run must execute *those* queries rather than a parallel set written for the
test, because a second set of queries is a second description of the same
business rule, and the two would drift the first time the rule changed. The
translation is therefore the whole contract: if the catalog SQL runs locally,
the local run is testing the production semantics.

Four constructs differ, and each has an exact translation:

``project.dataset.table``
    The local warehouse is a single SQLite file, so the dataset qualifier is
    dropped. A fully-qualified reference becomes a bare table name. This is the
    one lossy step, and it is safe precisely because the local warehouse has
    exactly one dataset: a qualifier that could disambiguate has nothing to
    disambiguate here.

``DATE_ADD(DATE(x), INTERVAL n DAY)``
    Becomes SQLite's modifier form, ``date(x, '+n days')``. The unit is carried
    through rather than assumed, so a month or year interval translates rather
    than being silently misread as days.

``CURRENT_DATE()`` and ``CURRENT_DATE``
    SQLite spells the current date as ``date('now')``.

Backtick quoting
    SQLite accepts double quotes for identifiers, which is standard SQL, so
    backticks become double quotes. Note the interaction with the house
    single-quote rule: these are *generated* SQL, not Python string literals, so
    the identifier delimiter is a SQL concern and is not a style violation.

The translator is deliberately not a general SQL parser. It recognises the
constructs this catalog uses and refuses anything it does not recognise, so a
query using an untranslated dialect feature fails loudly at translation time
instead of reaching SQLite and producing a wrong answer. Failing loudly is the
whole point: a silently mistranslated query returns plausible numbers.
"""

from __future__ import annotations

import re
from typing import Any

#: The opening of a ``DATE_ADD`` call. No single regular expression can delimit a
#: ``DATE_ADD`` call, because its inner expression may contain arbitrarily nested
#: balanced parentheses -- and that is exactly why the naive pattern failed here:
#: a lazy ``.+?`` matched the *outer* call first and read the interval out of the
#: *inner* one, so `DATE_ADD(DATE_ADD(d, INTERVAL 30 DAY), INTERVAL 58 DAY)`
#: became `date(DATE_ADD(d, '+30 days'), INTERVAL 58 DAY)` -- a statement SQLite
#: rejects, arrived at by silently mistranslating the query. The translator
#: therefore locates the opening here and finds the matching close by counting
#: brackets -- see :func:`_translate_date_adds`.
_DATE_ADD_OPEN = re.compile(r'DATE_ADD\s*\(', re.IGNORECASE)

#: The tail of a ``DATE_ADD`` call, matched at the position of the argument
#: separator. There is deliberately no ``^`` anchor: the pattern is applied with
#: ``match(sql, pos)``, where ``^`` asserts the start of the whole string rather
#: than the given offset, so an anchored pattern can never match a tail in the
#: middle of a statement.
_DATE_ADD_TAIL = re.compile(
  r'\s*,\s*INTERVAL\s+(?P<count>\d+)\s+'
  r'(?P<unit>DAY|DAYOFYEAR|HOUR|MINUTE|SECOND|WEEK|MONTH|QUARTER|YEAR)S?\s*\)',
  re.IGNORECASE,
)

#: A backtick-quoted, dot-separated warehouse path. The angle brackets in the
#: catalog's placeholder form are ordinary characters once quoted.
_QUALIFIED = re.compile(r'`[^`]*`')

#: Units translated to SQLite's modifier suffixes. SQLite understands only a
#: subset of the SQL interval units, so an unlisted unit is a translation error
#: rather than an approximation.
_UNIT_SUFFIX = {
  'DAY': 'days',
  'DAYOFYEAR': 'days',
  'WEEK': 'days',
  'HOUR': 'hours',
  'MINUTE': 'minutes',
  'SECOND': 'seconds',
  'MONTH': 'months',
  'QUARTER': 'months',
  'YEAR': 'years',
}

#: Constructs that have no SQLite equivalent and must not be translated by
#: guesswork. Their presence in a local query is an error.
_UNSUPPORTED = (
  'EXTERNAL_QUERY',
  'EXPORT DATA',
  'ML.PREDICT',
  'CREATE TEMP FUNCTION',
  'ASSERT',
)

#: Multipliers converting a translated unit into the next larger one. A unit
#: listed here has no SQLite modifier of its own, so it is expressed as a
#: multiple of a unit that does. This is the part that is easy to get wrong: a
#: translator that simply mapped every unit to a suffix would render ``INTERVAL
#: 2 WEEK`` as two *days*, which is a query that runs and returns the wrong date.
_UNIT_MULTIPLIER = {'QUARTER': 3, 'WEEK': 7}


class SqlTranslationError(ValueError):
  """Raised when a query uses a construct the translator will not guess at."""


def to_sqlite(sql: str) -> str:
  """Translate a warehouse query into the SQLite dialect.

  Args:
    sql: The query as written for the production warehouse.

  Returns:
    An equivalent SQLite query.

  Raises:
    SqlTranslationError: If the query uses a construct that has no exact
      translation. The error names the construct so the query can be rewritten
      rather than silently mis-executed.
  """
  upper = sql.upper()
  for construct in _UNSUPPORTED:
    if construct in upper:
      raise SqlTranslationError(
        f'{construct!r} has no local equivalent. Rewrite the query so the local '
        f'warehouse can express the same semantics without it.'
      )

  translated = _translate_date_adds(sql)

  translated = _QUALIFIED.sub(lambda m: _identifier(m.group(0)), translated)
  translated = re.sub(r'\bCURRENT_DATE\s*\(\s*\)', "date('now')", translated, flags=re.IGNORECASE)
  translated = re.sub(r'\bCURRENT_DATE\b(?!\s*\()', "date('now')", translated, flags=re.IGNORECASE)
  # An IFNULL with no arguments would be rewritten to a form that still has none;
  # leaving it alone preserves SQLite's own error, which names the function.
  return ' '.join(translated.split())


def _split_call(sql: str, open_index: int) -> tuple[str, re.Match[str], int]:
  """Split a ``DATE_ADD`` call into its inner expression and its interval tail.

  The call is delimited by counting brackets rather than by pattern, because the
  inner expression may contain arbitrarily nested parentheses -- including another
  ``DATE_ADD`` -- and a comma belonging to a nested call must not be mistaken for
  the separator.

  Args:
    sql: The statement being scanned.
    open_index: The index of the parenthesis opening the call.

  Returns:
    A three-tuple of the inner expression, the match describing the interval
    tail, and the index just past the call's closing parenthesis.

  Raises:
    SqlTranslationError: If the brackets never balance, or the interval does not
      parse. Both mean the expression is shaped in a way this translator does not
      understand, and guessing at it is what this module exists to avoid.
  """
  depth = 0
  separator: int | None = None
  position = open_index
  while position < len(sql):
    character = sql[position]
    if character == '(':
      depth += 1
    elif character == ')':
      depth -= 1
      if depth == 0:
        break
    elif character == ',' and depth == 1 and separator is None:
      separator = position
    position += 1
  else:
    raise SqlTranslationError(
      'Unbalanced parentheses in a DATE_ADD expression; the statement cannot be translated safely.'
    )

  if separator is None:
    raise SqlTranslationError(
      f'The DATE_ADD call opening at position {open_index} has no argument separator; expected '
      'the form DATE_ADD(<expression>, INTERVAL <n> <unit>).'
    )
  tail = _DATE_ADD_TAIL.match(sql, separator)
  if tail is None:
    raise SqlTranslationError(
      f'Could not parse the interval of the DATE_ADD call opening at position {open_index}. '
      'Expected the form DATE_ADD(<expression>, INTERVAL <n> <unit>).'
    )
  return sql[open_index + 1 : separator].strip(), tail, position + 1


def _translate_date_adds(sql: str) -> str:
  """Rewrite every ``DATE_ADD`` call, innermost first.

  Each pass rewrites exactly one call -- the last opening in the string, which is
  the innermost when the calls nest -- and repeats until none remain. Rewriting
  the inner call first is what lets the outer call wrap an already-translated
  expression rather than a raw warehouse one.

  Args:
    sql: The statement being translated.

  Returns:
    The statement with every ``DATE_ADD`` rewritten.

  Raises:
    SqlTranslationError: If a ``DATE_ADD`` call cannot be parsed, which means the
      expression is shaped in a way this translator does not understand.
  """
  translated = sql
  while True:
    opening = None
    for candidate in _DATE_ADD_OPEN.finditer(translated):
      opening = candidate
    if opening is None:
      return translated

    inner, tail, resume = _split_call(translated, opening.end() - 1)
    replacement = _date_add_replacement(inner, tail)
    translated = translated[: opening.start()] + replacement + translated[resume:]


def _date_add_replacement(inner: str, tail: re.Match[str]) -> str:
  """Build the SQLite form of one ``DATE_ADD`` call.

  Args:
    inner: The call's inner expression.
    tail: The match describing its ``INTERVAL`` tail.

  Returns:
    The replacement text for that call.

  Raises:
    SqlTranslationError: If the interval unit has no SQLite modifier.
  """
  unit = tail.group('unit').upper()
  count = int(tail.group('count')) * _UNIT_MULTIPLIER.get(unit, 1)
  if unit not in _UNIT_SUFFIX:
    raise SqlTranslationError(
      f'Interval unit {unit!r} has no SQLite equivalent; refusing to guess a duration.'
    )
  if count == 0:
    return f'date({inner})'
  return f"date({inner}, '{count:+d} {_UNIT_SUFFIX[unit]}')"


def _identifier(quoted: str) -> str:
  """Convert a backtick-quoted warehouse path into a SQLite identifier.

  Args:
    quoted: A backtick-quoted reference, possibly dot-separated.

  Returns:
    A double-quoted bare table name, with any embedded double quote doubled.
  """
  path = quoted.strip('`')
  table = path.rsplit('.', 1)[-1]
  return '"' + table.replace('"', '""') + '"'


def resolve_parameters(sql: str, parameters: dict[str, Any], globals_: dict[str, Any]) -> str:
  """Expand ``${section:key}`` and ``${globals:key}`` references in a query.

  The production catalog uses OmegaConf interpolation. A local run resolves the
  same references from plain dictionaries, so a query that is missing a
  parameter fails here -- with the key named -- rather than executing with a
  literal ``${...}`` in the SQL.

  Args:
    sql: The query, possibly containing references.
    parameters: The parameters document.
    globals_: The globals document.

  Returns:
    The query with every reference expanded.

  Raises:
    KeyError: If a reference names a key that is absent from its document.
  """
  def expand(match: re.Match[str]) -> str:
    section, _, key = match.group(1).partition(':')
    document = globals_ if section == 'globals' else parameters.get(section, {})
    if key not in document:
      raise KeyError(
        f'Query references ${{{match.group(1)}}} but {section!r} has no key {key!r}. '
        f'Available: {sorted(document)}'
      )
    return str(document[key])

  return re.sub(r'\$\{([^}]+)\}', expand, sql)
