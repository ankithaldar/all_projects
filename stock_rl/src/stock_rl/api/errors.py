#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The three ways this API refuses a request.

``BadRequest`` is the caller's fault, ``Refused`` is nobody's fault at
all, and ``ProviderFailed`` is an engineer's. They are three types rather
than one error carrying three messages because
:func:`stock_rl.api.dispatch` catches them in that order and maps each to
a different status code: telling a caller to change a request that was
never the problem is a false statement, and telling an operator that a
crash was a refusal hides the crash.

Split out of the former single-module ``stock_rl.api`` without change.
The three class bodies and every word of their docstrings are the text
that shipped, because the reasoning about which status code means what
is the part of this file most worth keeping intact.
'''

from __future__ import annotations


class BadRequest(ValueError):
  '''The request could not be understood or was not safe to run.

  Reported as HTTP 400. It covers malformed JSON, a non-object body, an
  unrecognised field and an out-of-range value. All four are client
  errors, and all four are worth a 400 rather than a 500 because a
  traceback in an HTTP response is a leak, not a diagnosis.
  '''


class Refused(ValueError):
  '''The request was well-formed but cannot be run right now.

  Reported as HTTP 422. An empty panel set is the common case: the
  server is healthy, the request was valid, and there is nothing to
  backtest. That is not the same as being asked for something invalid,
  and it is not a server fault either.
  '''


class ProviderFailed(RuntimeError):
  '''A weight provider raised instead of returning weights.

  Reported as HTTP 500, deliberately distinct from :class:`Refused`. A
  422 says "change your request"; a 500 says "this needs an engineer".
  Collapsing the two would have a caller retry a request that was never
  the problem.

  The distinction is enforced at the call site by :func:`_guarded`, which
  wraps the provider, rather than by inspecting the exception where it is
  caught. :func:`stock_rl.portfolio.run_portfolio` raises ``ValueError``
  for its own refusals and lets a provider's exception through
  unchanged, so the two are indistinguishable by type at the catch site;
  a provider raising ``ValueError`` would be reported to the caller as a
  422, which tells them to edit a request that was never the problem.

  PONYTAIL: the catch around a provider is deliberately broad. A
  provider is a callable owned by another module, and one that raises
  anything at all must not take down the endpoint whose whole purpose is
  to measure every provider. Ceiling: the failure is reported, not
  diagnosed -- the message carries the exception repr. Upgrade path:
  narrow the catch to the provider's own error type once ``baselines.py``
  declares one.
  '''
