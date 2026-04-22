from __future__ import annotations

import numpy as np
import pytest

from src.core.items import ItemId
from src.core.target_provider import TargetProvider


class TestTargetProvider:
    def test_load(self, target_provider: TargetProvider):
        targets = target_provider.targets
        assert ItemId.GOLD in targets
        assert targets[ItemId.GOLD] == 14

    def test_deliver(self, target_provider: TargetProvider):
        target_provider.deliver(ItemId.GOLD, 5)
        assert target_provider.remaining(ItemId.GOLD) == 9

    def test_deliver_over_target(self, target_provider: TargetProvider):
        target_provider.deliver(ItemId.GOLD, 20)
        assert target_provider.remaining(ItemId.GOLD) == 0

    def test_remaining_non_target_item(self, target_provider: TargetProvider):
        assert target_provider.remaining(ItemId.COTTON) == 0

    def test_is_complete_initially_false(self, target_provider: TargetProvider):
        assert target_provider.is_complete() is False

    def test_is_complete_after_all_delivered(self, target_provider: TargetProvider):
        for item_id, count in target_provider.targets.items():
            target_provider.deliver(item_id, count)
        assert target_provider.is_complete() is True

    def test_fraction_complete(self, target_provider: TargetProvider):
        assert target_provider.fraction_complete() == 0.0

    def test_fraction_complete_partial(self, target_provider: TargetProvider):
        total = sum(target_provider.targets.values())
        target_provider.deliver(ItemId.GOLD, 14)
        expected = 14 / total
        assert target_provider.fraction_complete() == pytest.approx(expected)

    def test_targets_remaining_array(self, target_provider: TargetProvider):
        arr = target_provider.targets_remaining_array()
        assert arr.shape == (23,)
        assert arr[int(ItemId.GOLD)] == 14

    def test_targets_total_array(self, target_provider: TargetProvider):
        arr = target_provider.targets_total_array()
        assert arr.shape == (23,)
        assert arr[int(ItemId.GOLD)] == 14
        assert arr[int(ItemId.COTTON)] == 0

    def test_reset(self, target_provider: TargetProvider):
        target_provider.deliver(ItemId.GOLD, 10)
        target_provider.reset()
        assert target_provider.remaining(ItemId.GOLD) == 14

    def test_deliver_non_target_no_effect(self, target_provider: TargetProvider):
        target_provider.deliver(ItemId.COTTON, 100)
        assert target_provider.fraction_complete() == 0.0

    def test_simple_targets(self, simple_targets_path: str):
        tp = TargetProvider(simple_targets_path)
        assert tp.targets[ItemId.STRING] == 5
        tp.deliver(ItemId.STRING, 5)
        assert tp.is_complete() is True
