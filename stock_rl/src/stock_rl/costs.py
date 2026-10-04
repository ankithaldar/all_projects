#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Transaction cost model for NSE cash equities.

Indian transaction costs are not one percentage. A single round trip pays
brokerage, STT, exchange transaction charges, SEBI turnover fees, stamp
duty (buy side only), GST (on brokerage + exchange + SEBI, not on STT),
DP charges (sell side only) and slippage. Each has a different side and a
different base. Getting this wrong by even 10bpper round trip dominates
the difference between a good and a mediocre strategy.

Every rate is a field on ``CostModel`` so a strategy can be priced under
several assumptions without touching this module.
'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ['Side', 'CostModel', 'DELIVERY', 'INTRADAY']


class Side(Enum):
  '''Side of a fill. Costs are asymmetric between buy and sell.'''

  BUY = 'buy'
  SELL = 'sell'


# A rate card is inherently wide, and every field is genuinely
# independent configuration rather than a responsibility. Splitting these
# into nested Brokerage/Taxes/Fee objects would add indirection and
# ceremony for no gain, so the attribute count is accepted deliberately.
# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True, slots=True)
class CostModel:
  '''Rates as fractions of traded notional, plus fixed rupee charges.

  Attributes:
    brokerage_pct: Broker commission rate. Delivery is zero only for a
      retail individual at a discount broker; institutional entity types
      and some brokers are charged 0.1% or Rs 20.
    brokerage_cap: Per-order rupee cap on brokerage, e.g. Zerodha's Rs 20.
    stt_buy: Securities transaction tax, buy side.
    stt_sell: Securities transaction tax, sell side.
    exchange_pct: NSE transaction charge, buy and sell. The client pays
      Rs 307 per crore (0.00307%) in total: since the March 2026
      reclassification this is Rs 306.99 as transaction charge plus
      Rs 0.01 as the NSE IPFT contribution. The older Rs 297 line item
      understates the real all-in charge by 3.3 percent.
    sebi_pct: SEBI turnover fee rate (buy and sell).
    stamp_duty_buy: Stamp duty, buy side only, never charged on sell.
    gst_pct: GST applied to brokerage + exchange + SEBI fees. Deliberately
      not applied to STT or stamp duty, which are themselves taxes.
    dp_charge: Rupee DP charge per sell. See the note on daily capping
      below; brokers quote Rs 15 to Rs 25 inclusive of GST.
    slippage_bps: Half-spread plus impact, applied to both sides.
  '''

  brokerage_pct: float = 0.0
  brokerage_cap: float = 20.0
  stt_buy: float = 0.001
  stt_sell: float = 0.001
  exchange_pct: float = 0.0000307
  sebi_pct: float = 0.000001
  stamp_duty_buy: float = 0.00015
  gst_pct: float = 0.18
  dp_charge: float = 15.34
  slippage_bps: float = 5.0

  def commission(self, side: Side, notional: float) -> float:
    '''Return brokerage for one order, honouring the rupee cap.

    Args:
      side: Side of the order. Delivery brokerage is symmetric, but the
        cap still applies per order rather than per side.
      notional: Absolute traded value in rupees.

    Returns:
      Brokerage in rupees, the lesser of the rate and the cap.
    '''
    del side
    return min(notional * self.brokerage_pct, self.brokerage_cap)

  def taxes_and_fees(self, side: Side, notional: float) -> float:
    '''Return all non-brokerage charges for one order.

    A zero notional means no order was placed, so no charge of any kind
    applies -- including the flat DP fee. This matters because the engine
    calls this on every rebalance, including the ones that close a
    position, and an unconditional per-order fee would bleed silently on
    every flat bar.

    Args:
      side: Side of the order. Stamp duty and DP charges are one-sided.
      notional: Absolute traded value in rupees.

    Returns:
      Total charges in rupees, excluding brokerage and slippage.
    '''
    if notional <= 0.0:
      return 0.0
    stt_rate = self.stt_sell if side is Side.SELL else self.stt_buy
    stt = notional * stt_rate
    exchange = notional * self.exchange_pct
    sebi = notional * self.sebi_pct
    stamp_rate = self.stamp_duty_buy if side is Side.BUY else 0.0
    stamp = notional * stamp_rate
    dp = self.dp_charge if side is Side.SELL else 0.0
    fee_base = self.commission(side, notional) + exchange + sebi
    gst = self.gst_pct * fee_base
    return stt + exchange + sebi + stamp + dp + gst

  def slippage(self, notional: float) -> float:
    '''Return slippage cost for one order.

    Args:
      notional: Absolute traded value in rupees.

    Returns:
      Slippage in rupees.
    '''
    return notional * self.slippage_bps / 10_000.0

  def round_trip(self, notional: float) -> float:
    '''Return the full cost of buying then selling ``notional``.

    This is the number a strategy author should sanity-check against, so
    it sums one buy and one sell at the same notional.

    Args:
      notional: Position size in rupees.

    Returns:
      Total round trip cost in rupees.
    '''
    buy = self.one_way(Side.BUY, notional)
    sell = self.one_way(Side.SELL, notional)
    return buy + sell

  def one_way(self, side: Side, notional: float) -> float:
    '''Return the full cost of a single order.

    Args:
      side: Side of the order.
      notional: Absolute traded value in rupees.

    Returns:
      Total cost in rupees, all components.
    '''
    brokerage = self.commission(side, notional)
    charges = self.taxes_and_fees(side, notional)
    return brokerage + charges + self.slippage(notional)


#: Delivery-equity defaults for a retail individual at a discount broker.
#: Brokerage is waived only for that case: Zerodha and Groww charge HUF,
#: trust, partnership and LLP accounts 0.1% or Rs 20 on delivery.
#
# PONYTAIL: these defaults are a dated snapshot verified against primary
# sources -- NSE's investor levies page (updated 17 Apr 2026), circular
# NSE/FA/73061 of 27 Feb 2026 for the exchange-charge reclassification,
# and SEBI (Stock Brokers) Regulations 2026 reg. 41 for the turnover fee.
# Two known ceilings remain. (1) DP charges are charged per DAY per scrip,
# not per order, so a strategy that sells one symbol twice in a session is
# over-charged here; on daily bars the two coincide, so per-order is kept
# and the intraday case is left to a future caller. (2) STT is levied on
# the side's VWAP across a settlement day, not per fill, which is why
# splitting one large order into slices is modelled as additive here and
# will slightly overstate STT. Upgrade path for both: load a dated rate
# card and a per-session accumulator. The dataclass is already the whole
# interface, so neither fix requires touching the arithmetic.
DELIVERY = CostModel()

#: Intraday defaults. Cheaper STT, lower stamp duty, no DP charge, but
#: brokerage is NOT free: discount brokers charge 0.03% per executed
#: order capped at Rs 20 (Zerodha, Dhan; Groww 0.05%, Angel 0.1%,
#: Upstox flat Rs 20). Leaving this at zero understates intraday cost by
#: roughly 60 percent, because brokerage then dominates a 3.55 bps levy.
INTRADAY = CostModel(
  brokerage_pct=0.0003,
  stt_buy=0.0,
  stt_sell=0.00025,
  stamp_duty_buy=0.00003,
  dp_charge=0.0,
)
