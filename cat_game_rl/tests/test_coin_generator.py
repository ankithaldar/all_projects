from __future__ import annotations

import pytest

from src.core.coin_generator import CoinGenerator


class TestCoinGenerator:
    def test_initial_balance(self):
        cg = CoinGenerator(500)
        assert cg.balance == 500

    def test_default_initial(self):
        cg = CoinGenerator()
        assert cg.balance == 0

    def test_tick(self):
        cg = CoinGenerator(0)
        cg.tick()
        assert cg.balance == 210

    def test_multiple_ticks(self):
        cg = CoinGenerator(0)
        for _ in range(10):
            cg.tick()
        assert cg.balance == 2100

    def test_spend_success(self):
        cg = CoinGenerator(1000)
        assert cg.spend(500) is True
        assert cg.balance == 500

    def test_spend_exact(self):
        cg = CoinGenerator(500)
        assert cg.spend(500) is True
        assert cg.balance == 0

    def test_spend_insufficient(self):
        cg = CoinGenerator(100)
        assert cg.spend(200) is False
        assert cg.balance == 100

    def test_spend_negative_raises(self):
        cg = CoinGenerator(100)
        with pytest.raises(ValueError):
            cg.spend(-50)

    def test_add(self):
        cg = CoinGenerator(100)
        cg.add(50)
        assert cg.balance == 150

    def test_add_negative_raises(self):
        cg = CoinGenerator(100)
        with pytest.raises(ValueError):
            cg.add(-50)

    def test_reset(self):
        cg = CoinGenerator(1000)
        cg.spend(500)
        cg.reset(0)
        assert cg.balance == 0

    def test_reset_with_value(self):
        cg = CoinGenerator(0)
        cg.reset(999)
        assert cg.balance == 999

    def test_coins_per_tick_constant(self):
        assert CoinGenerator.COINS_PER_TICK == 210
