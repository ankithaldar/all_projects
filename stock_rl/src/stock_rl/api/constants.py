#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The values this API commits to, declared once and read everywhere.

Every constant below is a decision rather than a default: the address the
server binds, the size of a body it will accept, the window the momentum
ranking is measured over, the keyword a weight provider takes its cap on,
the confidence the VaR is quoted at, and the disclaimer every payload
carries. They were interleaved with the code that used them when this was
one module, which meant a reader looking for the loopback address had to
know which of five sections to read.

Nothing else changed in the move. Each constant keeps its name, its value
and the ``#:`` comment that documents why it holds the number it holds,
and the loopback reasoning those comments refer to now lives in the
package docstring at :mod:`stock_rl.api`.
'''

from __future__ import annotations


#: Interface the API binds to. Loopback only, and the check is enforced
#: in :func:`is_loopback` rather than left to the operator's discipline.
#: See the module docstring for why this surface serves trading
#: decisions and audit state.
default_host = '127.0.0.1'

#: Default port. Not a privileged port, so the server can run unprivileged.
default_port = 8765

#: Largest accepted request body, 1 MiB. A backtest trigger is a few
#: hundred bytes; anything larger is a mistake or an attack, and reading
#: it would let one client pin the process.
max_body_bytes = 1 << 20

#: How much of an over-sized body is read and thrown away so the client
#: can finish writing and then read the refusal. Separate from
#: :data:`max_body_bytes` on purpose: the limit says what is accepted,
#: the drain says how politely the rejection is delivered. Answering 413
#: while the peer is still writing drops the connection mid-handshake,
#: and the peer sees a reset instead of the answer.
max_drain_bytes = 8 << 20

#: Bars in the trailing momentum window. 252 NSE trading days, matching
#: :data:`stock_rl.metrics.TRADING_DAYS_PER_YEAR`.
signal_lookback = 252

#: Most recent bars excluded from the momentum window. The 12-1
#: construction of Jegadeesh and Titman, applied because Sehgal and
#: Balakrishnan (2002) found Indian short-horizon returns continue rather
#: than reverse, so an unskipped lookback mixes two opposing effects.
signal_skip = 21

#: Symbols the momentum signal holds when the ranking is usable.
signal_top = 10

#: Cap on any single symbol's weight, matching ``env.nse.MAX_WEIGHT``.
signal_max_weight = 0.10

#: The keyword a weight provider takes its per-symbol cap on.
#:
#: It is bound by **keyword**, never positionally, and that is not a
#: style preference. The five providers in :mod:`stock_rl.baselines`
#: disagree about where the cap sits in their own signatures: second for
#: ``equal_weight`` and ``buy_and_hold``, third for ``low_volatility``,
#: fifth for ``momentum_ranked`` and sixth for
#: ``trend_filtered_momentum``. One positional call over that table hands
#: the weight cap to a moving-average window and computes
#: ``sma(closes, 0.1)`` instead of raising anything a caller can read.
#: ``pipeline._measure_engines`` documents the same hazard where it binds
#: the same cap.
cap_keyword = 'max_weight'

#: Confidence reported as the historical one-period VaR level.
var_confidence = 0.95

#: Weight difference below which a rebalance is not worth sending.
weight_epsilon = 1e-6

#: Attached to every payload, because a screen that renders Sharpe next
#: to a number without this line is an advertisement.
not_advice = (
  'Research output from a backtested model. Not financial advice, not a '
  'recommendation, and not a solicitation to buy or sell anything.'
)
