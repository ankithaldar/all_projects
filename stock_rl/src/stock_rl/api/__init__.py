#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Read-only HTTP API over the backtest core, standard library only.

Why there is no web framework here
----------------------------------

The design doc proposed FastAPI on uvicorn for inference and risk. This
module refuses both, for reasons that are about correctness rather than
taste.

One: ``http.server.ThreadingHTTPServer`` and ``BaseHTTPRequestHandler``
are already in the standard library, and this API is seven JSON
endpoints plus a document, over structures that already exist in this
repository. A framework would buy schema validation on the wire format;
:func:`stock_rl.api.validate._parse_request` does that by hand in about
sixty lines, and the hand-written version can be stricter than a coerced
one -- an unrecognised key is rejected here, because a silently ignored
``max_weight`` is exactly how a backtest ends up reporting performance it
did not earn.

Two: the project carries zero runtime dependencies as a rule rather than
an accident (see ``stock_rl/__init__.py``). A web framework would be the
largest dependency in the repository by an order of magnitude, in a
process that already computes Sharpe ratios by hand.

Three, and decisively: the project's own research concludes that risk and
execution must not be separated by a network boundary, because a
partition between them must fail either open -- unguarded orders reach
the exchange -- or closed, and neither is acceptable. Every additional
process is another boundary that can partition. One process has no such
failure mode by construction, and this module is the same process.

What this API is
----------------

An observation surface, plus one trigger that runs a backtest. It places
no orders, contacts no broker, and can neither trip nor clear the kill
switch. That asymmetry is deliberate. The design doc's kill switch is a
React button, and SEBI CIR/MRD/DP/09/2012 requires a switch that "is
expected to automatically trigger a halt on trading activity based on
pre-defined conditions". A button is a thing an operator has to think to
press, and the failure it exists for is the moment nobody is looking, so
no route here can halt or resume trading.
:mod:`stock_rl.risk.killswitch` is the control; this module reads its
state and cannot write it.

Known gap: no authentication, by deferral rather than by oversight
-------------------------------------------------------------------

There is no auth, no token and no TLS on this server, and that is a real
gap, not a considered posture. Everything the endpoints return is
readable by anyone who can open a socket to the port, which for a
trading desk means the book, the positions and the audit-relevant risk
state.

It is deferred for three specific reasons, and each would be a bad
reason on its own:

  1. Loopback binding is enforced, not merely defaulted (see
     :data:`stock_rl.api.default_host` and
     :func:`stock_rl.api.is_loopback`), so the exposure is "anything
     that can already run code as this user".
  2. SEBI requires the strategy servers to be located in India with "no
     interlink with any system or ID located/linked outside India", so a
     remote identity provider would have to be an Indian one, and picking
     one is a decision for whoever owns the deployment rather than for a
     library module.
  3. An in-process check that nobody has deployed is not a control. A
     half-built auth layer that fails open is worse than an honest
     absence, because it reads as protection.

What replaces it, before this ever listens on anything but loopback: put
a TLS-terminating reverse proxy in front, require a client certificate,
and have the proxy set the Algo ID header that
:mod:`stock_rl.compliance.algo_tag` already exists to record. Until that
exists, the correct reading of this module is "local developer tool", and
``GET /api/health`` says so in its ``advisory`` field.

Signals, honestly labelled
--------------------------

``GET /api/signals`` reports a momentum construction and nothing else,
because the other four families are measured rather than asserted: see
``GET /api/baselines``, which runs every provider in
:mod:`stock_rl.baselines` on the identical backtester, costs and splits.
That endpoint exists so the RL claim is checkable rather than asserted,
and a signal endpoint that quietly omitted the control arm would make it
un-checkable.

The ``confidence`` field is an ordinal midrank percentile of the
cross-sectional momentum score. It is not a calibrated probability and
must not be presented as one: nothing here has been fitted to outcome
frequency, so a "0.9 confidence" means "ranked near the top of this
panel", which is a much weaker statement than "90 percent likely to work".

Every JSON body this module emits carries
:data:`stock_rl.api.not_advice`, error responses included. A disclaimer
attached to the page but not to the payload stops being a disclaimer the
moment someone reads the payload.

How it is split
---------------

The public surface is unchanged: same names in :data:`__all__`, same
defaults, same status codes, same payload keys. The implementation moved
into the modules below.

``api.errors``
  The three refusals and the status code each maps to.
``api.protocols``
  The structural types and the injected seams, declared rather than
  imported.
``api.constants``
  Every declared value, including the loopback address and the
  disclaimer, with the ``#:`` comments that document them.
``api.models``
  :class:`Signal`, :class:`BacktestRequest`, :class:`PortfolioState` and
  the control arm, plus the coercions that fill them from an untrusted
  result.
``api.signals``
  The 12-1 momentum construction and its honest confidence.
``api.risk``
  The historical VaR and the read-only views of the kill switch.
``api.runner``
  The control arm's provider table, the weight-cap binding, and the
  default backtest runner.
``api.service``
  :class:`ApiService`, the state every read endpoint answers from.
``api.validate``
  Request validation, shared by the body and the query string.
``api.response``
  :class:`Response`, the routing table, and :func:`dispatch`.
``api.server``
  The socket policy, the HTTP framing, and the entry points.

One seam needs explaining, because it is not a thing a package normally
does. ``stock_rl.api`` used to be one namespace, so writing a name into
it changed what the code inside it did; that is how ``api.sharpe_ratio``,
``api.strategies``, ``api.index_html``, ``api.build_server`` and
``api.serve`` are seams the suite uses to prove the endpoints delegate
rather than reimplement. A package namespace is a copy of its
submodules', so the module type installed below pushes writes back down
into whichever submodule holds the name, and the single-namespace
behaviour the monolith had is preserved. The stdlib and ``typing`` names
the old module happened to import -- ``api.json``, ``api.time`` and the
rest -- are not re-exported, because no importer and no test reaches
them; the project names that were reachable are.
'''

from __future__ import annotations

import sys
from types import ModuleType

from stock_rl.api import constants as _constants
from stock_rl.api import errors as _errors
from stock_rl.api import models as _models
from stock_rl.api import protocols as _protocols
from stock_rl.api import response as _response
from stock_rl.api import risk as _risk
from stock_rl.api import runner as _runner
from stock_rl.api import server as _server
from stock_rl.api import service as _service
from stock_rl.api import signals as _signals
from stock_rl.api import validate as _validate
from stock_rl.api.constants import (
  cap_keyword,
  default_host,
  default_port,
  max_body_bytes,
  max_drain_bytes,
  not_advice,
  signal_lookback,
  signal_max_weight,
  signal_skip,
  signal_top,
  var_confidence,
  weight_epsilon,
)
from stock_rl.api.errors import BadRequest, ProviderFailed, Refused
from stock_rl.api.models import (
  BacktestRequest,
  PortfolioState,
  Signal,
  control_arm,
)
from stock_rl.api.protocols import (
  BacktestRunner,
  Clock,
  KillSwitchLike,
  SignalSource,
  ThresholdsLike,
  TripLike,
  WallClock,
  WeightProvider,
)
from stock_rl.api.response import Response, dispatch, routes
from stock_rl.api.risk import historical_var
from stock_rl.api.runner import (
  # Reachable by name because the suite checks it to prove the cap is
  # asked about rather than assumed. Deliberately not in ``__all__``.
  _accepts_cap,  # pylint: disable=unused-import
  run_baseline,
  strategies,
)
from stock_rl.api.server import (
  ApiHandler,
  ApiServer,
  ApiServer6,
  admitted_origin,
  bound_address,
  build_server,
  is_loopback,
  load_panels,
  main,
  serve,
)
from stock_rl.api.service import ApiService
from stock_rl.api.signals import momentum_signals

# Names the old module imported and a caller could reach through it. They
# are re-exported so ``api.sharpe_ratio`` and ``api.index_html`` still
# resolve -- and, through the module type below, still replace what the
# code calls.
from stock_rl import __version__, baselines
from stock_rl.api.response import index_html, script, stylesheet
from stock_rl.bars import Bar, load_csv
from stock_rl.indicators import momentum
from stock_rl.metrics import max_drawdown, sharpe_ratio, total_return
from stock_rl.portfolio import PortfolioResult, run_portfolio

__all__ = [
  'ApiHandler',
  'ApiServer',
  'ApiServer6',
  'ApiService',
  'BacktestRequest',
  'BacktestRunner',
  'BadRequest',
  'PortfolioState',
  'ProviderFailed',
  'Refused',
  'Response',
  'Signal',
  'SignalSource',
  'admitted_origin',
  'bound_address',
  'build_server',
  'control_arm',
  'default_host',
  'default_port',
  'dispatch',
  'historical_var',
  'is_loopback',
  'load_panels',
  'main',
  'momentum_signals',
  'not_advice',
  'routes',
  'run_baseline',
  'serve',
  'strategies',
]

# Read by :meth:`_Surface.__setattr__`.
_SUBMODULES = (
  _constants,
  _errors,
  _models,
  _protocols,
  _response,
  _risk,
  _runner,
  _server,
  _service,
  _signals,
  _validate,
)


class _Surface(ModuleType):
  '''The package's module type, which forwards writes to the submodules.

  A package namespace is built out of ``from`` statements, so every name
  in it is a copy of one a submodule holds. Setting
  ``stock_rl.api.sharpe_ratio`` would rebind only the copy and leave
  :meth:`stock_rl.api.service.ApiService.equity` calling the original, so
  the seams the suite uses to prove the endpoints delegate --
  ``sharpe_ratio``, ``max_drawdown``, ``strategies``, ``index_html``,
  ``build_server``, ``serve`` -- would stop being seams.

  Writing through to the submodules restores what the single module had:
  one namespace, so one assignment, and the code inside it sees that
  assignment. Nothing about how a response is built, a status code is
  chosen or an exception is worded depends on this.

  Note:
    This is the same ``sys.modules[__name__].__class__`` swap
    :mod:`stock_rl.context.fuse` uses, for the same reason: a rename that
    would otherwise be a breaking change. It is the only magic in the
    package, and it is here so the split is a split.
  '''

  def __setattr__(self, name, value):
    '''Bind ``name`` here and in every submodule that holds it.

    Args:
      name: Attribute being written.
      value: Value to bind.

    Returns:
      None.
    '''
    super().__setattr__(name, value)
    for module in globals().get('_SUBMODULES', ()):
      if name in vars(module):
        setattr(module, name, value)


# ``python -m stock_rl.api`` is served by :mod:`stock_rl.api.__main__`,
# which is the only way a package is directly executable.
sys.modules[__name__].__class__ = _Surface
