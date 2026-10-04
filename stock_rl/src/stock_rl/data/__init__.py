#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Market data acquisition: the trading calendar and the vendor feed.

Two concerns that have to exist before a single bar can be ingested.

``calendar``
  Which dates trade. NSE equity cash runs 09:15 to 15:30 IST, and
  Muhurat trading is a separate session with separately announced hours.
  Holiday lists are **injected, never bundled**: an embedded list goes
  stale silently and a wrong holiday is a day of wrong returns.

``vendors``
  Market data comes from a licensed vendor, not from scraping NSE. TrueData
  and Global Financial Datafeeds are both on NSE's authorised real-time
  list; entitlements are per segment (CM, F&O, CD, Debt, Index) rather
  than a boolean; TradingView Inc. is on that list too, which surprises
  people who assume it is a free public source. Vendor names are not
  hardcoded here, because the authoritative list changes without notice
  and lives at nseindia.com/market-data/real-time-data-subscription.

``ReplayVendor`` is the shipped implementation: a feed served from bars
already in memory. That is what makes ingestion testable offline, and it
is not a mock -- it enforces the same contract a licensed adapter does.
'''

from stock_rl.data.calendar import (
  Session,
  TradingCalendar,
  TradingDay,
  coerce_date,
  muhurat_session,
  nse_normal_session,
  weekend_days,
)
from stock_rl.data.vendors import (
  Entitlement,
  EntitlementError,
  MarketDataFeed,
  ReplayVendor,
  Segment,
)

__all__ = [
  'Entitlement',
  'EntitlementError',
  'MarketDataFeed',
  'ReplayVendor',
  'Segment',
  'Session',
  'TradingCalendar',
  'TradingDay',
  'coerce_date',
  'muhurat_session',
  'nse_normal_session',
  'weekend_days',
]
