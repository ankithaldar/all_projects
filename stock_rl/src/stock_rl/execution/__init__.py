#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Sizing, broker abstraction and the append-only decision log.

Three modules, in the order an order passes through them:

``sizing``
  Turns an edge and a haircut into a quantity. It **re-uses** the
  pre-trade checker from :mod:`stock_rl.risk.checks` rather than holding
  limits of its own, because a sizer with private caps and an RMS with
  exchange caps is two sets of numbers that agree until the day they do
  not -- and on that day the larger one wins silently. A size the RMS
  will not clear raises rather than being shrunk to fit, because a
  quietly halved order hides the fact that the strategy wanted more than
  the account could give, and that fact is the signal the operator needs.

``broker``
  A ``Broker`` Protocol with two honest implementations and no vendor.
  :class:`~stock_rl.execution.broker.PaperBroker` records what would have
  been sent and never leaves the process;
  :class:`~stock_rl.execution.broker.NullBroker` refuses everything so
  that "no broker configured" is a rejection with a reason rather than
  an ``AttributeError`` three layers up.
  :class:`~stock_rl.execution.broker.FailoverRouter` models multi-broker
  failover in process. **There is no API key, no network call and no real
  broker integration in this package, by design**: SEBI requires the
  exchange's prior permission for each algo before a broker may offer the
  facility, this project has not been registered, and NSE 9.1/9.9 require
  fresh registration for *any* change to the logic governing a black-box
  algo, which a nightly-retrained model is, every night.

``audit``
  The append-only JSONL decision log with a hash chain. Its job is to
  answer "is the running process the registered one", which is what
  discharges the re-registration obligation, and its honest limitation is
  that per-decision rationale is **not required by any Indian rule**. It
  is an internal control justified as strategy-identity evidence. Retention
  is 8 years, from SEBI (Stock Brokers) Regulations 2026 reg. 16, above
  the 5-year NSE para 10.3 floor; there is no sourced 7-year figure.

The dependency arrow points one way: this package imports
:mod:`stock_rl.risk` and nothing in :mod:`stock_rl.risk` imports this
package. **Risk runs before execution, never after**, and the import
graph is what makes that structural rather than a matter of discipline.
'''

from stock_rl.execution.audit import (
  AuditChainError,
  AuditLog,
  AuditRecord,
  ChainVerification,
  audit_schema,
  decision_log_class,
  feature_hash,
  genesis_hash,
  verify_chain,
)
from stock_rl.execution.broker import (
  Broker,
  BrokerAck,
  BrokerError,
  BrokerHealth,
  FailoverRouter,
  NullBroker,
  PaperBroker,
  PaperOrder,
  cancelled,
  filled,
  open_order,
  order_statuses,
  rejected,
)
from stock_rl.execution.sizing import (
  RiskCapBreach,
  SizingInputs,
  SizingResult,
  StockSizer,
  kelly_fraction,
  kelly_method,
  max_fraction,
  uncertainty_adjusted_size,
  vol_target_fraction,
  vol_target_method,
)

__all__ = [
  'AuditChainError',
  'AuditLog',
  'AuditRecord',
  'Broker',
  'BrokerAck',
  'BrokerError',
  'BrokerHealth',
  'ChainVerification',
  'FailoverRouter',
  'NullBroker',
  'PaperBroker',
  'PaperOrder',
  'RiskCapBreach',
  'SizingInputs',
  'SizingResult',
  'StockSizer',
  'audit_schema',
  'cancelled',
  'decision_log_class',
  'feature_hash',
  'filled',
  'genesis_hash',
  'kelly_fraction',
  'kelly_method',
  'max_fraction',
  'open_order',
  'order_statuses',
  'rejected',
  'uncertainty_adjusted_size',
  'verify_chain',
  'vol_target_fraction',
  'vol_target_method',
]
