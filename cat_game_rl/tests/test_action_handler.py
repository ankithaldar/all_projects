from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.items import ItemId, CraftingTree, CRAFTABLE_ITEM_IDS, NUM_CRAFTABLE
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.core.cost_calculator import CostCalculator
from src.env.action_handler import ActionHandler


class TestDecode:
    def test_all_zeros(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
        decoded = handler.decode(action)
        assert len(decoded) == 0

    def test_single_item(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
        action[0] = 5  # STRING
        decoded = handler.decode(action)
        assert decoded[ItemId.STRING] == 5

    def test_multiple_items(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        action = np.zeros(NUM_CRAFTABLE, dtype=np.int64)
        action[0] = 3  # STRING
        action[1] = 2  # WOOD
        decoded = handler.decode(action)
        assert len(decoded) == 2
        assert decoded[ItemId.STRING] == 3
        assert decoded[ItemId.WOOD] == 2


class TestValidateAndApply:
    def test_valid_action(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        stash.add(ItemId.COTTON, 30)
        coins = CoinGenerator(10000)
        slots = SlotScheduler(crafting_tree)

        decoded = {ItemId.STRING: 5}
        result = handler.validate_and_apply(decoded, stash, coins, slots)

        assert ItemId.STRING in result["applied"]
        assert result["applied"][ItemId.STRING] == 5
        assert slots.is_busy(ItemId.STRING)
        assert stash.get(ItemId.COTTON) == 30 - 3 * 5

    def test_busy_slot_rejected(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        stash.add(ItemId.COTTON, 60)
        coins = CoinGenerator(10000)
        slots = SlotScheduler(crafting_tree)

        slots.start(ItemId.STRING, 1)
        decoded = {ItemId.STRING: 5}
        result = handler.validate_and_apply(decoded, stash, coins, slots)

        assert ItemId.STRING in result["rejected"]
        assert result["rejected"][ItemId.STRING] == "slot_busy"

    def test_insufficient_materials_clipped(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        stash.add(ItemId.COTTON, 9)  # only enough for 3
        coins = CoinGenerator(10000)
        slots = SlotScheduler(crafting_tree)

        decoded = {ItemId.STRING: 5}
        result = handler.validate_and_apply(decoded, stash, coins, slots)

        assert result["applied"][ItemId.STRING] == 3

    def test_insufficient_coins_clipped(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        stash.add(ItemId.COTTON, 300)
        coins = CoinGenerator(100)
        slots = SlotScheduler(crafting_tree)

        decoded = {ItemId.STRING: 10}
        result = handler.validate_and_apply(decoded, stash, coins, slots)

        applied_batch = result["applied"].get(ItemId.STRING, 0)
        assert applied_batch > 0
        assert applied_batch <= 10

    def test_no_materials_rejected(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        coins = CoinGenerator(10000)
        slots = SlotScheduler(crafting_tree)

        decoded = {ItemId.STRING: 5}
        result = handler.validate_and_apply(decoded, stash, coins, slots)

        assert ItemId.STRING in result["rejected"]

    def test_coins_deducted(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        stash.add(ItemId.COTTON, 30)
        coins = CoinGenerator(10000)
        slots = SlotScheduler(crafting_tree)

        decoded = {ItemId.STRING: 5}
        handler.validate_and_apply(decoded, stash, coins, slots)

        expected_cost = math.ceil(CostCalculator.total_cost(50, 5))
        assert coins.balance == 10000 - expected_cost


class TestActionMask:
    def test_zero_always_valid(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        coins = CoinGenerator(0)
        slots = SlotScheduler(crafting_tree)

        mask = handler.compute_action_mask(stash, coins, slots)
        for i in range(NUM_CRAFTABLE):
            assert mask[i * 21] is True or mask[i * 21] == True

    def test_busy_slot_all_masked(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        stash.add(ItemId.COTTON, 100)
        coins = CoinGenerator(10000)
        slots = SlotScheduler(crafting_tree)
        slots.start(ItemId.STRING, 1)

        mask = handler.compute_action_mask(stash, coins, slots)
        for b in range(1, 21):
            assert mask[0 * 21 + b] == False

    def test_with_resources_unmasked(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        stash.add(ItemId.COTTON, 30)
        coins = CoinGenerator(10000)
        slots = SlotScheduler(crafting_tree)

        mask = handler.compute_action_mask(stash, coins, slots)
        assert mask[0 * 21 + 1] == True  # batch=1 for STRING
        assert mask[0 * 21 + 10] == True  # batch=10 for STRING

    def test_mask_shape(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        coins = CoinGenerator(0)
        slots = SlotScheduler(crafting_tree)

        mask = handler.compute_action_mask(stash, coins, slots)
        assert mask.shape == (NUM_CRAFTABLE * 21,)
        assert mask.dtype == np.bool_
