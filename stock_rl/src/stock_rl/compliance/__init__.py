#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Data acquisition compliance controls.

Four concerns, deliberately kept in separate modules because they rest on
four different bases, and conflating them is how a compliance register
ends up full of fabricated citations:

``robots``
  robots.txt, which has **no legal force in India**. No Indian court has
  decided the issue. It is respected as risk management, and because a
  site's Terms of Use frequently incorporates it by reference -- and a
  Terms of Use *is* an enforceable contract.

``ratelimit``
  Request pacing. The familiar "one request per second" has **no Indian
  legal source**; it is engineering risk management against the IT Act
  s.43(e) and (f) disruption exposure, and the module says so.

``retention``
  Audit record retention per record class, each figure carrying the
  instrument it came from. SEBI's algo circular specifies no period at
  all; 5 years is the NSE operational-modalities para 10.3 floor, 8
  years is SEBI (Stock Brokers) Regulations 2026 Reg. 16, and there is no
  sourced 7-year figure.

``algo_tag``
  NSE order tagging: the 15-digit NNF ID, its 13th-digit algo flag, and
  the Algo ID pairing rule from NSE/FAOP/69296.

The Algo ID validators live in ``retention`` rather than in their own
module because an Algo ID is an audit-trail field, but they are
re-exported here so a caller wiring an order path finds them in one
place.
'''

from stock_rl.compliance.algo_tag import (
  AlgoTagError,
  algo_flag_digit,
  allows_algo,
  build_nnf_id,
  client_direct_api_platform,
  is_algo_flag,
  platform_of,
  validate_algo_tag,
)
from stock_rl.compliance.ratelimit import (
  BackoffPolicy,
  RateLimiter,
  parse_retry_after,
)
from stock_rl.compliance.retention import (
  AlgoIdError,
  AuditRecordClass,
  expiry_date,
  is_expired,
  registered_algo_ids,
  require_registration,
  retention_classes,
  retention_table,
  retention_years,
  unregistered_algo_id,
  validate_algo_id,
)
from stock_rl.compliance.robots import RobotsChecker, RobotsUnavailable

__all__ = [
  'AlgoIdError',
  'AlgoTagError',
  'AuditRecordClass',
  'BackoffPolicy',
  'RateLimiter',
  'RobotsChecker',
  'RobotsUnavailable',
  'algo_flag_digit',
  'allows_algo',
  'build_nnf_id',
  'client_direct_api_platform',
  'expiry_date',
  'is_algo_flag',
  'is_expired',
  'parse_retry_after',
  'platform_of',
  'registered_algo_ids',
  'require_registration',
  'retention_classes',
  'retention_table',
  'retention_years',
  'unregistered_algo_id',
  'validate_algo_id',
  'validate_algo_tag',
]
