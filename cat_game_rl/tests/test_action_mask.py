from __future__ import annotations

import numpy as np
import pytest

from src.core.items import ItemId, CraftingTree, NUM_CRAFTABLE
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.env.action_handler import ActionHandler


class TestActionMaskIntegration:
    def test_no_resources_only_zeros_valid(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        coins = CoinGenerator(0)
        slots = SlotScheduler(crafting_tree)

        mask = handler.compute_action_mask(stash, coins, slots)

        for i in range(NUM_CRAFTABLE):
            assert mask[i * 21] == True
            for b in range(1, 21):
                assert mask[i * 21 + b] == False

    def test_one_item_affordable(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        stash.add(ItemId.COTTON, 6)  # enough for 2 string
        coins = CoinGenerator(300)  # enough for several
        slots = SlotScheduler(crafting_tree)

        mask = handler.compute_action_mask(stash, coins, slots)

        string_idx = 0
        assert mask[string_idx * 21 + 1] == True
        assert mask[string_idx * 21 + 2] == True
        assert mask[string_idx * 21 + 3] == False  # only 6 cotton / 3 = 2

    def test_coin_limited(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        stash.add(ItemId.COTTON, 300)
        coins = CoinGenerator(50)  # only enough for 1
        slots = SlotScheduler(crafting_tree)

        mask = handler.compute_action_mask(stash, coins, slots)

        string_idx = 0
        assert mask[string_idx * 21 + 1] == True
        assert mask[string_idx * 21 + 2] == False

    def test_mask_valid_then_action_succeeds(self, crafting_tree: CraftingTree):
        handler = ActionHandler(crafting_tree, max_batch=20)
        stash = Stash()
        stash.add(ItemId.COTTON, 15)
        coins = CoinGenerator(5000)
        slots = SlotScheduler(crafting_tree)

        mask = handler.compute_action_mask(stash, coins, slots)

        valid_batches = []
        string_idx = 0
        for b in range(1, 21):
            if mask[string_idx * 21 + b]:
                valid_batches.append(b)

        assert len(valid_batches) > 0

        decoded = {ItemId.STRING: valid_batches[-1]}
        result = handler.validate_and_apply(decoded, stash, coins, slots)
        assert ItemId.STRING in result["applied"]
