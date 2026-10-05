#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Structured logging for this project, and nothing else.

**Why this file exists.** Nothing in the backtest core, the RL
environments, the risk layer or the execution layer emitted anything. A
run that produced a wrong number gave no clue why, so the diagnosis was
reading code -- which is how this repo found that a momentum function
computed the *negative* of momentum, and that a partial weight mapping
reads as an exit in one engine and as "hold" in another. A line of
output at the moment of the decision is worth more than an afternoon of
reading, so the facility comes first and the instrumentation follows it.

**What this is.** Four things and no more: one :func:`configure` that
owns handlers and levels, :func:`get_logger` for a namespace, a
:func:`debug_step` context manager that brackets an operation with its
inputs and elapsed time, and :func:`log_event` for greppable
``key=value`` output. It is deliberately not a framework. There is no
plugin registry, no async path, no rotation and no file handler, because
every one of those is a decision that belongs to whoever eventually
runs this in production and none of them is needed to find out why a
backtest printed the wrong Sharpe.

**Logging must never take down a backtest.** That is the load-bearing
property here, stronger than tidiness. A log line that raises in the
middle of ``run_portfolio`` turns a wrong number into a lost run, which
is strictly worse: the wrong number was at least reproducible. So every
path from a caller's value to a log record is wrapped, an object whose
``__str__`` raises is recorded as ``<unrepresentable>`` rather than
propagated, and a handler that itself raises is swallowed. The same
reasoning is why :func:`log_event` returns a bool: a caller in a hot
loop can assert the suppression without capturing anything, and a test
can tell "did no work" apart from "did work and failed".

**Logging must not touch the caller's data.** Fields are copied before
they are redacted and rendered, because the caller passed them to a log
call and still needs them: the weight mapping being logged is the same
object ``_Book`` is about to trade against, and a logging call that
rewrote a nested dict would corrupt the run it was describing. Copies
are plain ``dict`` and ``list``, so a ``defaultdict`` in a field comes
back as a ``dict``.

**Zero cost when disabled.** The first thing every entry point does is
``logger.isEnabledFor(level)``, and below that threshold no dict is
copied, no value is rendered and no clock is read. ``Step`` is a plain
class rather than a ``@contextlib.contextmanager`` generator for the
same reason: a generator-based context manager constructs a generator
and a throwaway object on every ``with``, and both boundaries here can
short-circuit on a single boolean. The residual cost of the enabled
path is one level lookup and, for a caller that cannot hoist the check,
one ``**fields`` dict built by the interpreter before this module is
entered. A caller in the innermost loop of ``run_portfolio`` should
guard with ``if _log.isEnabledFor(logging.DEBUG):`` itself; that is
documented here because it is the one instruction that cannot be
enforced from inside the callee.

**Redaction is the part that has to be right**, because a log file is
the one artefact that outlives the process, gets copied into tickets and
gets pasted into chat. What this project actually handles, read from
:mod:`stock_rl.compliance.algo_tag`, :mod:`stock_rl.compliance.retention`
and :mod:`stock_rl.execution.broker`:

  * **Algo IDs.** ``algo_id`` on an :class:`~stock_rl.risk.checks.OrderRequest`
    is an exchange-registered, venue-namespaced string. Its two sentinels
    (``0`` and ``99999``) are public, but a real ID is an opaque
    registration and is redacted.
  * **NNF IDs and platform prefixes.** 15 digits; a CTCL prefix's first
    six digits are the client PIN, per this project's own reading of
    :func:`~stock_rl.compliance.algo_tag.allows_algo`.
  * **Client codes and account identifiers** -- ``client_code``,
    ``user_id``, ``dp_id``, PAN, demat. There is no broker integration
    in this build, so no such value is produced here yet; the rule exists
    now because the first one that reaches a log line is the moment the
    rule would be written, and that is too late.
  * **Credentials** -- API keys and secrets, tokens, cookies, PINs,
    passwords, TOTP seeds. Same reasoning: :mod:`stock_rl.execution.broker`
    refuses to hold a credential, and the redaction is not to be defeated
    by the day someone adds one.
  * **Broker and exchange order ids.** A venue-assigned identifier is a
    handle into somebody else's system and is not ours to publish.

Two rules do the work, and both fail closed:

  * **by key name** -- an exact set plus a set of substrings, matched
    case-insensitively with ``-`` folded to ``_``. The split is
    deliberate: ``pin`` is an *exact* key only, because as a substring it
    matches ``mapping``; ``pass`` likewise matches ``bypass``. Exact keys
    are checked first and substrings are chosen to be unambiguous.
  * **by value shape** -- any string with a run of nine or more digits is
    redacted regardless of the key it arrived under. That catches an NNF
    ID or a 12-digit platform prefix logged under an innocuous field
    name, which is the failure a key-based rule cannot see coming, and it
    over-redacts a large integer statistic. The ceiling is stated in the
    note on :data:`LONG_DIGIT_RUN`.

A key that cannot be turned into text is redacted without being read.

Callers extend the rules by passing ``extra_keys=`` to
:func:`log_event`, :func:`redact` or :func:`redact_fields`. There is no
global mutable registry on purpose: a process-wide rule that any module
can widen is a rule nobody can audit, and the whole point of this list is
that it can be read.

**Greppable, not prose.** Output is ``event=<name> key=value ...``, one
line, values quoted with :func:`shlex.quote` so that ``shlex.split``
parses the line back into the same fields. Prose is what made the
diagnosis an afternoon of reading; a field is what makes it a grep.

**Deliberately not done.** Exception *messages* are not logged by
:func:`debug_step`, only exception *types* -- see the note on
:meth:`Step.__exit__` -- because a message is free text that could carry
anything, and this module cannot redact text it does not know how to
parse.
'''

from __future__ import annotations

import logging
import os
import re
import shlex
import time
from collections.abc import Mapping

__all__ = [
  'CYCLE',
  'DEFAULT_FORMAT',
  'DEFAULT_LEVEL',
  'LEVEL_ENV_VAR',
  'LONG_DIGIT_RUN',
  'REDACTED',
  'ROOT_LOGGER_NAME',
  'SENSITIVE_KEYS',
  'SENSITIVE_SUBSTRINGS',
  'UNRENDERABLE',
  'Fields',
  'Step',
  'configure',
  'debug_step',
  'get_logger',
  'is_sensitive',
  'log_event',
  'redact',
  'redact_fields',
]

#: Root of this project's logger namespace. Everything this module hands
#: out sits under it, so ``stock_rl`` is the one level to configure and a
#: caller never has to know where the root of the namespace lives.
ROOT_LOGGER_NAME = 'stock_rl'

#: Attribute stamped on every handler :func:`configure` installs, and the
#: only thing it will ever remove. Repeated calls must not double every
#: line, and a handler this module did not install is not this module's
#: to take away.
HANDLER_TAG = 'stock_rl.logging_support'

#: Level used when the caller names none and the environment is silent.
#: INFO, because a silent default at WARNING hides exactly the wrong
#: things -- which panel ran, which symbol was skipped -- and this
#: project's default mode is a paper backtest, where volume is low.
DEFAULT_LEVEL = 'INFO'

#: Environment variable consulted when :func:`configure` is given no
#: level. Read at call time rather than at import time so a test, or a
#: wrapper that sets the variable before dispatch, gets what it asked for.
LEVEL_ENV_VAR = 'STOCK_RL_LOG_LEVEL'

#: Default format. Includes the logger name because the question "which
#: layer produced this" is the first one asked of a new line, and asctime
#: because a run whose numbers changed between two invocations is
#: otherwise undatable.
DEFAULT_FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'

#: What a redacted value is replaced with. Bracketed so it cannot be
#: mistaken for a value the caller actually logged, and because a
#: grep for a credential and a grep for this marker answer different
#: questions.
REDACTED = '<redacted>'

#: What a value that cannot be rendered at all becomes. Distinct from
#: :data:`REDACTED` on purpose: "we removed this on purpose" and "this
#: object refused to be printed" are different findings, and collapsing
#: them would hide a broken ``__str__`` behind a security control that
#: looked like it was working.
UNRENDERABLE = '<unrepresentable>'

#: Marker for a value shortened at :data:`MAX_VALUE_CHARS`, and for a
#: container nested deeper than the depth limit. Plain ASCII so that a
#: grep for it does not depend on the terminal's encoding.
TRUNCATED = '...'

#: What a container that contains itself becomes. A distinct marker
#: because "this structure loops" is a fact about the data and
#: :data:`TRUNCATED` is a fact about this module's patience; reporting
#: both as an ellipsis would lose the difference a caller can act on.
CYCLE = '<cycle>'

#: Longest single value in a line, before quoting. 240 characters is
#: longer than any metric, symbol or order reason this project produces,
#: and short enough that a line stays readable in a terminal and does not
#: wrap in a CI log viewer.
# ponytail: truncation is by character count, so a container is cut at an
# arbitrary point rather than at an item boundary, and the value has
# already been redacted in full by then, so nothing is half-redacted.
# Ceiling: a 5000-character payload cannot be read back from the log.
# Upgrade path: emit a reference to the whole value -- a path, a digest --
# instead of a prefix, which needs somewhere to put it and so needs a
# decision this project has not made.
MAX_VALUE_CHARS = 240

#: Most fields per line. A log line carrying fifty fields is a dump, not
#: a record, and the ones dropped are counted in the ``dropped=`` field
#: rather than vanishing silently.
# ponytail: the cap is a fixed count and the dropped fields are whichever
# sort last by name, not whichever matter least -- there is no way to
# express "keep these five" from a keyword argument. Ceiling: a wide
# payload silently loses its alphabetically-last fields. Upgrade path: a
# caller-supplied priority, which needs an ordering hint in the field name
# and is not worth building before a caller needs it.
MAX_FIELDS = 32

#: How deep :func:`redact` walks a container before giving up. Three
#: levels covers the shapes this project logs -- a weight mapping, a list
#: of order reasons, a mapping of symbol to position -- and stops a
#: self-referential structure from recursing forever.
MAX_DEPTH = 3

#: Digit-run length at which a *value* is redacted whatever its key says.
#: Nine is the shortest run that cannot be a rupee amount, a weight, a
#: price or a timestamp this project writes, and the longest that cannot
#: be an NSE field of interest: a platform prefix is 12 digits and an NNF
#: ID is 15.
# ponytail: the threshold is a fixed length, not a per-field rule, so a
# genuinely large integer statistic -- a nine-digit quantity, a
# nanosecond timestamp -- is redacted too. That is the safe direction to
# be wrong in, and a false redaction is visible in the output while a
# false non-redaction is not. Upgrade path: name the handful of fields
# allowed to carry long digit runs in an explicit allowlist checked before
# this rule, which becomes worth doing the first time a log line loses a
# number somebody needed.
LONG_DIGIT_RUN = 9

#: Keys whose value is a credential or an account identifier, matched
#: exactly after normalisation. Anything that would be a false positive as
#: a substring lives here rather than below: ``pin`` is a substring of
#: ``mapping`` and ``pass`` of ``bypass``, which is why neither appears
#: in :data:`SENSITIVE_SUBSTRINGS`.
SENSITIVE_KEYS = frozenset({
  'access_token',
  'account_id',
  'account_number',
  'acct_no',
  'algo_id',
  'api_key',
  'api_secret',
  'api_secret_key',
  'apikey',
  'auth_token',
  'authorization',
  'bearer',
  'broker_order_id',
  'client_code',
  'client_id',
  'client_order_id',
  'clientcode',
  'clientid',
  'cookie',
  'cookies',
  'credential',
  'credentials',
  'demat',
  'dp_id',
  'exchange_order_id',
  'mpin',
  'nnf_id',
  'non_algo_id',
  'order_id',
  'otp',
  'pan',
  'pass',
  'passcode',
  'password',
  'passwd',
  'pin',
  'private_key',
  'pwd',
  'refresh_token',
  'request_id',
  'session_id',
  'totp',
  'unregistered_algo_id',
  'user_id',
  'userid',
  'vendor_order_id',
  'webhook_secret',
})

#: Fragments that mark a key as sensitive wherever they appear in it.
#: Every entry here is chosen to be unambiguous: each was checked against
#: this project's own vocabulary, which is where ``pin`` and ``pass``
#: failed. ``nnf`` matches ``nnf_id`` and ``algo_id`` matches an Algo ID
#: spelled out in full.
SENSITIVE_SUBSTRINGS = (
  'access_token',
  'account_id',
  'account_number',
  'algo_id',
  'api_key',
  'apikey',
  'auth_token',
  'client_code',
  'client_id',
  'credential',
  'nnf',
  'order_id',
  'password',
  'passwd',
  'private_key',
  'refresh_token',
  'secret',
  'user_id',
)

#: A field's rendered form. Fields arrive as ``**kwargs`` and are copied
#: into one of these before anything reads or rewrites them.
Fields = dict[str, object]

#: Compiled once because it runs on every string field of every enabled
#: log line. ``\d`` is Unicode-aware, so a non-ASCII digit run is caught
#: too; that is the intended direction.
_LONG_DIGITS = re.compile(rf'\d{{{LONG_DIGIT_RUN},}}')


def get_logger(name: str = '') -> logging.Logger:
  '''Return a logger namespaced under :data:`ROOT_LOGGER_NAME`.

  Args:
    name: Dotted suffix, or a full name already under the root. Both are
      accepted, because a caller holding ``__name__`` will pass it and a
      caller holding a short label should not have to remember the root.
      Empty returns the namespace logger itself.

  Returns:
    The logger. A ``Logger``, not a wrapper, so a caller can still call
    ``isEnabledFor`` to guard a hot loop and can hand it to anything in
    the standard library that wants a logger.
  '''
  if not name:
    return logging.getLogger(ROOT_LOGGER_NAME)
  if name == ROOT_LOGGER_NAME or name.startswith(f'{ROOT_LOGGER_NAME}.'):
    return logging.getLogger(name)
  return logging.getLogger(f'{ROOT_LOGGER_NAME}.{name}')


def configure(level: int | str | None = None,
              stream: object | None = None,
              fmt: str | None = None) -> logging.Logger:
  '''Install this project's handler, format and level. Idempotent.

  **Idempotent because the classic bug here is doubled output.** Every
  call removes the handlers it installed last time -- identified by
  :data:`HANDLER_TAG`, not by type -- and installs exactly one, so
  calling this twice from a library and a CLI emits one line. A handler
  this module did not install is left alone, on purpose: taking over a
  handler it did not create is how a host application's logging gets
  eaten by a library.

  The level is set on the ``stock_rl`` logger and the handler is attached
  to the **root** logger. That split is deliberate: a child logger with
  no level of its own resolves to ``stock_rl``'s, so this is the single
  threshold every project logger consults, while attaching at the root
  means records propagate to whatever else the process has already set
  up -- ``stock_rl.api``'s existing ``_log`` included, with no edit to
  that file.

  Args:
    level: Level name or numeric level. None reads :data:`LEVEL_ENV_VAR`
      from the environment, falling back to :data:`DEFAULT_LEVEL`.
    stream: Destination, passed to :class:`logging.StreamHandler`. None
      means stderr, which is what that class does with no stream.
    fmt: Format string, or None for :data:`DEFAULT_FORMAT`.

  Returns:
    The ``stock_rl`` logger, so a caller can configure and go on to use
    it in one expression.

  Raises:
    ValueError: If the level is not a known level name or an int. Loud,
      at start-up: a run that logs nothing because a flag was misspelled
      is a run that lies by omission.
  '''
  root = logging.getLogger()
  for handler in list(root.handlers):
    if getattr(handler, HANDLER_TAG, None) is not None:
      root.removeHandler(handler)
      handler.close()
  installed = logging.StreamHandler(stream)
  installed.setFormatter(logging.Formatter(fmt or DEFAULT_FORMAT))
  # ponytail: identify our own handler by an attribute stamped on it
  # rather than by comparing against a module-level registry of them.
  # Two libraries doing the same thing to the same root logger could
  # remove each other's handler, since the tag is a constant in each.
  # Ceiling: mutual destruction between two tagging libraries.
  # Upgrade path: include the owning module's __name__ in the tag, which
  # makes two instances of this same module the only collision left.
  setattr(installed, HANDLER_TAG, True)
  root.addHandler(installed)
  logger = logging.getLogger(ROOT_LOGGER_NAME)
  logger.setLevel(_resolve_level(level))
  return logger


def log_event(logger: logging.Logger, event: str, *,
              level: int = logging.INFO,
              extra_keys: tuple[str, ...] = (),
              **fields: object) -> bool:
  '''Emit one greppable ``event=... key=value`` line.

  Args:
    logger: Logger to write through, typically from :func:`get_logger`.
    event: Stable event name, snake_case. The one token a grep starts
      from, so it is a name rather than a sentence.
    level: Numeric level. A string is not accepted here, unlike in
      :func:`configure`: this is the hot path and a lookup per call is
      the wrong trade.
    extra_keys: Additional key names to treat as sensitive for this call
      only. The extension point for a field this project has not met
      yet; see the module docstring on why it is a parameter and not a
      global registry.
    **fields: Structured fields. Copies are taken, sensitive ones are
      redacted and the rest are rendered.

  Returns:
    True if a record was emitted, False if the level was disabled or
    anything at all went wrong. The return exists so a caller can
    distinguish "nothing to say" from "said nothing"; nothing is raised,
    ever, into a backtest.

  Raises:
    Nothing. One exception, and it is not this function's to catch: a
    field named ``level`` collides with the keyword-only ``level``
    argument and the interpreter raises the ``TypeError`` at the call
    site. No field in this project is named ``level``; a caller who needs
    one should call :func:`Step.set` or rename the field.

  # ponytail: ``level`` is keyword-only rather than positional, which is
  what makes the collision above possible. Ceiling: one reserved field
  name. Upgrade path: take the level out of the signature entirely and
  expose four thin wrappers -- one per level -- if a caller ever needs a
  field called ``level``, which nothing here does.
  '''
  try:
    if not logger.isEnabledFor(level):
      return False
    return _emit(logger, event, level, fields, extra_keys)
  except Exception:  # pylint: disable=broad-except
    # A handler that raises, a broken stream, an exotic level. The
    # contract is that a log call cannot take down the caller, and the
    # only honest place to honour that is here rather than in a dozen
    # individual try blocks.
    #
    # ponytail: the swallow is completely silent -- no record reaches any
    # sink, not even stderr, so a log call that fails every time is
    # invisible and a run can produce no output at all with nothing to say
    # why. Ceiling: a permanently broken logging setup is undetectable.
    # Upgrade path: count the failures on the logger and report the count
    # from configure(), which is one integer and one line and keeps the
    # promise that a log call never raises.
    return False


def is_sensitive(key: object,
                 extra_keys: tuple[str, ...] = ()) -> bool:
  '''Return whether a field name must be redacted.

  Args:
    key: Field name, normalised by case-folding and ``-`` to ``_``.
    extra_keys: Additional names, normalised the same way.

  Returns:
    True for an exact match in :data:`SENSITIVE_KEYS`, for any key
    containing a fragment from :data:`SENSITIVE_SUBSTRINGS`, and for a
    key that cannot be turned into text at all. The last one fails
    closed: a name this module cannot read is a name it cannot clear.
  '''
  try:
    name = _normalise(key)
  except Exception:  # pylint: disable=broad-except
    return True
  if name in SENSITIVE_KEYS:
    return True
  if any(fragment in name
         for fragment in map(_normalise, extra_keys)):
    return True
  return any(fragment in name for fragment in SENSITIVE_SUBSTRINGS)


def redact(key: object, value: object,
           extra_keys: tuple[str, ...] = ()) -> object:
  '''Return a redacted copy of one field.

  Args:
    key: Field name, tested by :func:`is_sensitive`.
    value: The caller's value.
    extra_keys: Additional sensitive names for this call.

  Returns:
    :data:`REDACTED` when the key is sensitive, the value itself when it
    is a scalar that needs nothing, or a **new** plain ``dict`` or
    ``list`` when it is a container. The caller's object is never
    modified and never handed on, because the caller still needs it.
  '''
  return _redact_entry(key, value, extra_keys, 0, ())


def redact_fields(fields: Mapping[str, object],
                  extra_keys: tuple[str, ...] = ()) -> Fields:
  '''Return a redacted copy of a whole mapping.

  Args:
    fields: The caller's fields.
    extra_keys: Additional sensitive names for this call.

  Returns:
    A new ``dict``. Key order is preserved, because the rendering step
    sorts anyway and preserving it here keeps a caller who inspects the
    result from seeing a surprising reordering.
  '''
  out: Fields = {}
  for key, value in fields.items():
    try:
      out[key] = redact(key, value, extra_keys)
    except Exception:  # pylint: disable=broad-except
      # One unreadable key must not cost the caller the rest of the line.
      out[key] = UNRENDERABLE
  return out


class Step:
  '''One timed operation, logged at both boundaries.

  A plain class rather than a ``@contextlib.contextmanager`` generator,
  because a generator context manager allocates a generator and a
  throwaway object for every ``with`` and this one is meant to be usable
  in a loop; both boundaries here can instead return on a single
  boolean when the level is disabled.

  The caller may add fields discovered while the body ran, through
  :meth:`set`, and they appear on the closing line rather than the
  opening one -- which is the point, since a result is not known when the
  step begins.

  Both boundaries merge their fields into one mapping and call the
  internal emitter rather than expanding ``**fields``. A caller field
  named ``step`` or ``elapsed_s`` would otherwise be a duplicate keyword
  argument, and the resulting ``TypeError`` would be raised inside
  ``__exit__`` where it would replace -- and so hide -- whatever exception
  the body raised. A merged dict has no such failure mode: the reserved
  names simply win, which is also the answer to what the caller's field
  should mean.

  Attributes:
    name: The operation name, echoed on both lines.
    logger: Where to write.
    level: Numeric level for both boundaries.
    fields: Inputs, logged on the opening line.
    extra_keys: Additional sensitive field names for this step.
  '''

  name: str
  logger: logging.Logger
  level: int

  def __init__(self, name: str, logger: logging.Logger, level: int,
               fields: Mapping[str, object],
               extra_keys: tuple[str, ...] = ()) -> None:
    '''Build a step.

    Args are the attributes; there are none beyond them.

    Args:
      name: Operation name.
      logger: Where to write.
      level: Numeric level.
      fields: Inputs logged on the opening line.
      extra_keys: Additional sensitive field names.
    '''
    self.name = name
    self.logger = logger
    self.level = level
    self.extra_keys = extra_keys
    # Held raw and redacted at each boundary rather than here. A disabled
    # step must copy nothing, and a step whose result is learned in the
    # body must redact that result once it exists.
    self.fields: Fields = dict(fields)
    self.added: Fields = {}
    self._started: float | None = None
    self._elapsed = 0.0

  @property
  def enabled(self) -> bool:
    '''Return whether this step will write anything.

    Returns:
      True when the level is enabled. A caller can test it once and skip
      building an expensive payload.
    '''
    return self.logger.isEnabledFor(self.level)

  @property
  def elapsed_s(self) -> float:
    '''Return seconds between the boundaries, never negative.

    Returns:
      The measured duration, or 0.0 for a step that never ran or has
      not finished. Read from :func:`time.perf_counter`, which is
      monotonic, so this cannot go backwards on a clock adjustment --
      a wall-clock subtraction could report a negative duration, which
      would be a lie about the operation rather than a measurement of it.
    '''
    return self._elapsed

  def set(self, **fields: object) -> None:
    '''Add fields to be logged on the closing line only.

    Args:
      **fields: Result fields, typically a size, a cost or a count.
        Held unredacted and redacted at :meth:`__exit__`, so a disabled
        step copies nothing at all; redaction on entry would charge a
        log call that emits nothing for the privilege of being quiet.
    '''
    self.added.update(fields)

  def __enter__(self) -> Step:
    '''Log the opening boundary and start the clock.

    Returns:
      This step, so a caller can reach :meth:`set` inside the body.
    '''
    if not self.enabled:
      return self
    self._started = time.perf_counter()
    _emit(self.logger, 'step.begin', self.level,
          {**self.fields, 'step': self.name}, self.extra_keys)
    return self

  def __exit__(self, exc_type: type[BaseException] | None,
               exc: BaseException | None,
               traceback: object | None) -> bool:
    '''Log the closing boundary and never swallow the body's exception.

    Only the exception **type** is logged, never its message. A message
    is free text this module has no way to redact -- it could be an
    ``api_key=... rejected`` string from a broker adapter -- and a
    facility whose redaction is the load-bearing property cannot be the
    one that leaks.

    Args:
      exc_type: Type of the exception raised in the body, if any.
      exc: The exception itself, unused and not logged, per above.
      traceback: The traceback, unused.

    Returns:
      False, always: a step that failed is a step whose caller must
      still see the failure.
    '''
    del exc, traceback
    if self._started is not None:
      self._elapsed = max(
        0.0, time.perf_counter() - self._started)
    if not self.enabled:
      return False
    closing: Fields = dict(self.added)
    closing.update({'step': self.name, 'elapsed_s': self._elapsed,
                    'ok': exc_type is None})
    if exc_type is not None:
      closing['error'] = exc_type.__name__
    _emit(self.logger, 'step.end', self.level, closing, self.extra_keys)
    return False


def debug_step(name: str, logger: logging.Logger | None = None, *,
               level: int = logging.DEBUG,
               extra_keys: tuple[str, ...] = (),
               **fields: object) -> Step:
  '''Return a context manager bracketing one named operation.

  Args:
    name: Operation name, echoed on both lines and greppable.
    logger: Where to write. None uses this module's own logger, which
      puts the caller's events under ``stock_rl.logging_support``. A
      caller that wants its events attributed to its own layer passes its
      logger, and a CLI that wires this up should.
    level: Numeric level. DEBUG by default, because a step boundary is
      the definition of a debug line.
    extra_keys: Additional sensitive field names for this step.
    **fields: Inputs, logged on the opening line. Not the results; those
      go through :meth:`Step.set` so they land on the closing line.

  Returns:
    A :class:`Step`, usable as a context manager.
  '''
  # ponytail: a step is not reentrant and carries no step id, so nesting
  # two of them over the same logger produces four lines with no way to
  # pair a begin with its end. Ceiling: nested steps are ambiguous in the
  # output. Upgrade path: a monotonic counter in the step name, which
  # costs a field on both boundaries and is only worth it once a real
  # caller nests.
  if logger is None:
    logger = get_logger(__name__)
  return Step(name, logger, level, fields, extra_keys)


def _resolve_level(level: int | str | None) -> int:
  '''Return a numeric level from a level, an environment variable or a default.

  Args:
    level: As in :func:`configure`.

  Returns:
    The numeric level.

  Raises:
    ValueError: If the value is not a known level name or a plain int.
      bool is excluded explicitly because it is an int and ``True``
      would otherwise configure the whole project at CRITICAL.
  '''
  if level is None:
    level = os.environ.get(LEVEL_ENV_VAR) or DEFAULT_LEVEL
  if isinstance(level, bool):
    raise ValueError(
      f'level must be a level name or an int, got {level!r}')
  if isinstance(level, int):
    return level
  name = str(level).strip().upper()
  known = logging.getLevelNamesMapping()
  if name not in known:
    raise ValueError(
      f'unknown log level {level!r}; known: {sorted(known)}')
  return known[name]


def _emit(logger: logging.Logger, event: str, level: int,
          fields: Mapping[str, object],
          extra_keys: tuple[str, ...] = ()) -> bool:
  '''Render one record and hand it to a logger, absorbing any failure.

  Args:
    logger: Logger to write through.
    event: Event name.
    level: Numeric level.
    fields: The caller's fields, unredacted.
    extra_keys: Additional sensitive field names.

  Returns:
    True once the record has been handed to the logger, False if anything
    at all went wrong. :func:`log_event` checks the level before coming
    here; this function is the total barrier, which is why every write
    path in the module goes through it.
  '''
  try:
    # '%s' with an argument rather than an interpolated message, so a
    # percent sign inside a caller-supplied value cannot be read as a
    # conversion by a formatter.
    logger.log(level, '%s', _message(event, fields, extra_keys))
  except Exception:  # pylint: disable=broad-except
    return False
  return True


def _message(event: str, fields: Mapping[str, object],
             extra_keys: tuple[str, ...]) -> str:
  '''Return the rendered line for one record.

  Args:
    event: Event name.
    fields: The caller's fields, unredacted.
    extra_keys: Additional sensitive field names.

  Returns:
    ``event=<name> key=value ...``, fields sorted by name so two runs of
    the same code produce diffable lines. Quoting is
    :func:`shlex.quote`, which means ``shlex.split`` parses a captured
    line back into the fields that went in.
  '''
  clean = redact_fields(fields, extra_keys)
  pairs = [f'event={_quote(_text(event))}']
  keys = sorted(clean, key=_text)
  for key in keys[:MAX_FIELDS]:
    pairs.append(f'{_quote(_text(key))}={_quote(_text(clean[key]))}')
  if len(keys) > MAX_FIELDS:
    # The count of what was dropped, because a field that silently
    # vanished is the failure this whole module exists to prevent, and a
    # caller seeing 'dropped=5' can go and split the call.
    pairs.append(f'dropped={len(keys) - MAX_FIELDS}')
  return ' '.join(pairs)


def _redact_entry(key: object, value: object, extra_keys: tuple[str, ...],
                  depth: int, seen: tuple[int, ...]) -> object:
  '''Redact one mapping entry, keeping the walk's depth and path.

  Exists because :func:`redact` starts a walk at depth zero, and a
  mapping value routed through it would reset the depth and forget the
  path for every nested level. That is not a cosmetic bug: a dict
  holding itself would be re-walked from the top each time, and the depth
  limit -- the only thing bounding the recursion -- would never be
  reached.

  Args:
    key: The entry's key, tested by :func:`is_sensitive`.
    value: The entry's value.
    extra_keys: Additional sensitive field names.
    depth: Nesting depth of this entry.
    seen: Container ``id``s already on this path.

  Returns:
    :data:`REDACTED` for a sensitive key, otherwise the value copied and
    redacted.
  '''
  if is_sensitive(key, extra_keys):
    return REDACTED
  return _redact_value(value, extra_keys, depth, seen)


def _redact_value(value: object, extra_keys: tuple[str, ...], depth: int,
                  seen: tuple[int, ...] = ()) -> object:
  '''Return a redacted copy of a container or a shape-checked scalar.

  Args:
    value: The value to copy.
    extra_keys: Additional sensitive field names, applied to mapping keys.
    depth: Current nesting depth.
    seen: ``id`` of every container on the current path, so a structure
      that contains itself is reported rather than walked forever.

  Returns:
    A plain ``dict`` or ``list`` copy for a container, the string
    :data:`REDACTED` for one whose shape betrays an identifier,
    :data:`CYCLE` for one already on this path, the value itself for any
    other scalar, or :data:`TRUNCATED` past :data:`MAX_DEPTH`.

  Raises:
    Nothing is raised out of this function on purpose; every caller is on
    a path that must not raise. An exception here becomes a line this
    module did not write, which is exactly the outcome the module exists
    to prevent.
  '''
  if depth >= MAX_DEPTH:
    # ponytail: a container deeper than three levels is reported as
    # truncated rather than walked. Ceiling: a legitimately nested
    # payload loses its tail and there is no count of what was lost.
    # Upgrade path: a depth marker in the rendered value carrying the
    # number of levels skipped, which is a one-line change here and a
    # parser change in every consumer.
    return TRUNCATED
  if isinstance(value, str):
    return REDACTED if _LONG_DIGITS.search(value) else value
  if not isinstance(value, (Mapping, list, tuple, set, frozenset)):
    # ponytail: a non-container is handed on untouched and rendered with
    # str/repr rather than copied, so a mutable object passed as a field
    # is rendered live and could in principle change between this return
    # and the handler's write. Ceiling: a mutable, repr-inconsistent
    # object is a value whose rendered form is not reproducible. Upgrade
    # path: copy known mutable types explicitly -- the dataclasses this
    # project actually logs -- and record the ones that cannot be copied
    # so an operator sees that rather than a plausible value.
    return value
  marker = id(value)
  if marker in seen:
    # ponytail: identity, not equality, is what decides a cycle, so two
    # equal-but-distinct containers are both rendered in full and only a
    # container that contains *itself* is reported. Ceiling: a payload
    # that shares structure rather than looping is walked twice, and two
    # equal large payloads cost the same as one of each. Upgrade path:
    # a value-equality check with a depth budget, which costs a comparison
    # per container and so is only worth it if such a payload shows up.
    #
    # A cycle is a fact about the value and is reported as one. Without
    # this the copy would recurse until ``repr`` raised, which the
    # unrepresentable marker would hide: a legible ``<cycle>`` beats a
    # swallowed RecursionError, because this module exists to be read.
    return CYCLE
  seen = (*seen, marker)
  if isinstance(value, Mapping):
    return {key: _redact_entry(key, item, extra_keys, depth + 1, seen)
            for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_redact_value(item, extra_keys, depth + 1, seen)
            for item in value]
  # A set, sorted so that it renders identically across runs. Sorting by
  # the rendered text rather than by the value itself, because a set may
  # hold mixed types and comparing them raises.
  return sorted((_redact_value(item, extra_keys, depth + 1, seen)
                 for item in value), key=_text)


def _normalise(name: object) -> str:
  '''Return a field name folded for comparison.

  Args:
    name: Any object, treated as text.

  Returns:
    The name lowercased with ``-`` folded to ``_``. Not ``strip``-ed of
    internal whitespace, because a name with a space in it is a name
    nobody meant to write and matching it exactly is the safe direction.
  '''
  return str(name).strip().lower().replace('-', '_')


def _text(value: object) -> str:
  '''Return a string for any value, or :data:`UNRENDERABLE`.

  Args:
    value: Anything a caller logged.

  Returns:
    The string, or :data:`UNRENDERABLE`. The fallback is what makes the
    "never raises into the caller" property testable: an object whose
    ``__str__`` raises produces this marker and the caller proceeds
    unaware. ``repr`` is tried after ``str`` for a plain object, because
    a dataclass with no ``__str__`` still has a useful ``repr``.
  '''
  try:
    if isinstance(value, str):
      return value
    return str(value)
  except Exception:  # pylint: disable=broad-except
    pass
  try:
    return repr(value)
  except Exception:  # pylint: disable=broad-except
    return UNRENDERABLE


def _quote(text: str) -> str:
  '''Return a value quoted for a line and shortened past the limit.

  Args:
    text: Already-rendered text.

  Returns:
    The text quoted by :func:`shlex.quote`, so ``shlex.split`` recovers
    it exactly, with :data:`TRUNCATED` appended when it was too long.
    Quoting is what lets a value contain a space or a quote: the reason
    this module does not hand-roll a quoting function is that getting the
    escaping right is the whole job, and the standard library does it.
  '''
  if len(text) > MAX_VALUE_CHARS:
    text = text[:MAX_VALUE_CHARS] + TRUNCATED
  return shlex.quote(text)
