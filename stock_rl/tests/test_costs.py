#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the NSE transaction cost model.'''

import pytest

from stock_rl.costs import DELIVERY, INTRADAY, CostModel, Side

RUPEE = 100_000.0


class TestCommission:
  '''Brokerage rate and per-order cap.'''

  def test_delivery_brokerage_is_waived(self):
    assert DELIVERY.commission(Side.BUY, RUPEE) == 0.0

  def test_cap_binds_on_small_orders(self):
    model = CostModel(brokerage_pct=0.0003)
    # 0.03% of 1000 is 0.3, which is under the 20 rupee cap.
    assert model.commission(Side.BUY, 1_000.0) == pytest.approx(0.3)

  def test_cap_binds_on_large_orders(self):
    model = CostModel(brokerage_pct=0.0003)
    # 0.03% of 1 crore is 30000, so the cap must win.
    assert model.commission(Side.BUY, 10_000_000.0) == 20.0


class TestOneSidedCharges:
  '''Stamp duty and DP charges must appear on exactly one side.'''

  def test_stamp_duty_is_buy_only(self):
    assert DELIVERY.stamp_duty_buy > 0.0
    single = CostModel(stamp_duty_buy=0.001, gst_pct=0.0,
                       exchange_pct=0.0, sebi_pct=0.0,
                       stt_buy=0.0, stt_sell=0.0, dp_charge=0.0)
    assert single.taxes_and_fees(Side.BUY, RUPEE) == pytest.approx(100.0)
    assert single.taxes_and_fees(Side.SELL, RUPEE) == 0.0

  def test_dp_charge_is_sell_only(self):
    single = CostModel(dp_charge=15.34, gst_pct=0.0, exchange_pct=0.0,
                       sebi_pct=0.0, stt_buy=0.0, stt_sell=0.0,
                       stamp_duty_buy=0.0)
    assert single.taxes_and_fees(Side.SELL, RUPEE) == pytest.approx(15.34)
    assert single.taxes_and_fees(Side.BUY, RUPEE) == 0.0


class TestGstBase:
  '''GST is charged on fees, not on STT.'''

  def test_gst_excludes_stt(self):
    only_stt = CostModel(stt_buy=0.001, stt_sell=0.0, exchange_pct=0.0,
                         sebi_pct=0.0, stamp_duty_buy=0.0,
                         gst_pct=0.18, dp_charge=0.0)
    # STT is 100 rupees; GST on top of it would make 118.
    assert only_stt.taxes_and_fees(Side.BUY, RUPEE) == pytest.approx(100.0)

  def test_gst_applies_to_exchange_and_sebi(self):
    model = CostModel(stt_buy=0.0, stt_sell=0.0, exchange_pct=0.001,
                      sebi_pct=0.0, stamp_duty_buy=0.0, gst_pct=0.18,
                      dp_charge=0.0)
    # 100 rupees exchange charge + 18% GST on it.
    assert model.taxes_and_fees(Side.BUY, RUPEE) == pytest.approx(118.0)


class TestAsymmetry:
  '''Delivery and intraday must not price identically.'''

  def test_intraday_is_cheaper_than_delivery(self):
    assert INTRADAY.round_trip(RUPEE) < DELIVERY.round_trip(RUPEE)

  def test_round_trip_is_symmetric_under_delivery_stt(self):
    # Delivery charges STT both sides, so buy+sell must roughly double
    # the buy-side tax figure.
    assert DELIVERY.round_trip(RUPEE) > 2 * DELIVERY.one_way(
      Side.BUY, RUPEE) * 0.9

  def test_sell_is_not_equal_to_buy_under_intraday(self):
    # On 1 lakh intraday, the sell side pays STT 25.00 and the buy side
    # pays stamp duty 3.00. Both pay the same exchange/SEBI/GST/slippage,
    # so the sides differ by exactly 22.00.
    difference = (INTRADAY.one_way(Side.SELL, RUPEE)
                  - INTRADAY.one_way(Side.BUY, RUPEE))
    assert difference == pytest.approx(22.0)

  def test_intraday_side_breakdown_in_rupees(self):
    # Pins the full arithmetic so a future rate change is visible as a
    # deliberate diff rather than a silent shift in reported Sharpe.
    exchange = RUPEE * INTRADAY.exchange_pct
    sebi = RUPEE * INTRADAY.sebi_pct
    gst = INTRADAY.gst_pct * (exchange + sebi)
    slippage = RUPEE * INTRADAY.slippage_bps / 10_000.0
    expected_buy = RUPEE * 0.00003 + exchange + sebi + gst + slippage
    expected_sell = RUPEE * 0.00025 + exchange + sebi + gst + slippage
    assert INTRADAY.one_way(Side.BUY, RUPEE) == pytest.approx(expected_buy)
    assert INTRADAY.one_way(Side.SELL, RUPEE) == pytest.approx(expected_sell)


class TestZeroNotional:
  '''A flat position must cost nothing, not a per-order minimum.'''

  def test_zero_notional_is_free(self):
    assert DELIVERY.one_way(Side.BUY, 0.0) == 0.0
    assert DELIVERY.one_way(Side.SELL, 0.0) == 0.0
    assert DELIVERY.round_trip(0.0) == 0.0

  def test_no_dp_charge_when_flat(self):
    # Regression guard: DP charge is a flat per-order fee, so it was
    # previously charged even at zero notional and bled 15.34 rupees on
    # every close-out.
    flat = CostModel(dp_charge=15.34, gst_pct=0.0, exchange_pct=0.0,
                     sebi_pct=0.0, stt_buy=0.0, stt_sell=0.0,
                     stamp_duty_buy=0.0)
    assert flat.taxes_and_fees(Side.SELL, 0.0) == 0.0
    assert flat.taxes_and_fees(Side.SELL, 1.0) == 15.34


class TestModelIsImmutable:
  '''Rate tables are shared module state; mutation would be a global bug.'''

  def test_fields_are_frozen(self):
    model = CostModel()
    with pytest.raises(AttributeError):
      model.stt_buy = 0.5  # type: ignore[misc]
