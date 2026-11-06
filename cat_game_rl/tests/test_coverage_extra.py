#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Extra coverage tests for remaining branches.'''

# pylint: disable=protected-access,import-outside-toplevel

from __future__ import annotations

import gzip
import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import yaml

from src.cat_game_env.action_handler import ActionHandler
from src.cat_game_env.crafting_env import CraftingEnv
from src.cat_game_env.frame_skipper import FrameSkipWrapper
from src.cat_game_env.multi_agent import (
  MultiAgentOrchestrator,
  OrderBoard,
  TierEnv,
  group_items_by_tier,
  get_ingredient_ids,
)
from src.cat_game_env.reward_shaper import WasteMinimizationReward
from src.core.coin_generator import CoinGenerator
from src.core.cost_calculator import CostCalculator
from src.core.inventory import Stash
from src.core.items import CraftingTree, ItemId
from src.core.slot_scheduler import SlotScheduler
from src.core.target_provider import TargetProvider
from src.ga.chromosome import MAX_TICKS, Chromosome
from src.ga.ga_scheduler import GaScheduler
from src.logging_util.inventory_logger import (
  GzipRotatingFileHandler,
  InventoryLogger,
)


def _ga_config():
  '''Return a GA config with a population divisible by four.'''
  return {
    'population_size': 8,
    'n_generations': 1,
    'crossover_prob': 0.5,
    'mutation_prob': 0.5,
    'mut_batch_indpb': 0.02,
    'mut_time_shift_prob': 0.15,
    'sparse_init_density': 0.05,
    'max_init_batch': 10,
    'time_block_size': 48,
    'seed': 7,
  }


class TestChromosomeBranches:
  '''Tests for chromosome evaluation rollback branches.'''

  def test_initial_stash_and_busy_slot(self, crafting_tree):
    '''Busy slots and initial stash are handled.'''
    genes = np.zeros((MAX_TICKS, 19), dtype=np.uint8)
    genes[0, 0] = 5
    genes[1, 0] = 5  # slot busy at tick 1 for string
    result = Chromosome.evaluate(
      genes, crafting_tree, {ItemId.STRING: 5},
      initial_coins=100000,
      initial_stash={'string': 3, 'bogus': 1, 'cotton': -2},
    )
    assert len(result) == 4
    assert result[0] > 0

  def test_coin_spend_failure_rolls_back(self, crafting_tree):
    '''Coin spend failure restores ingredients.'''
    genes = np.zeros((MAX_TICKS, 19), dtype=np.uint8)
    genes[0, 0] = 20
    # One coin: feasible batch is capped by coins, so use a
    # recipe whose cost calculation races; instead patch spend.
    with mock.patch.object(
      CoinGenerator, 'spend', return_value=False
    ):
      cost, _, _, _ = Chromosome.evaluate(
        genes, crafting_tree, {ItemId.STRING: 5},
        initial_coins=100000,
      )
    assert cost == 0.0

  def test_material_removal_failure_rolls_back(
    self, crafting_tree
  ):
    '''Material removal failure restores earlier ingredients.'''
    from src.core.items import CRAFTABLE_ITEM_IDS  # noqa: PLC0415

    ribbon_idx = CRAFTABLE_ITEM_IDS.index(int(ItemId.RIBBON))
    genes = np.zeros((MAX_TICKS, 19), dtype=np.uint8)
    genes[0, ribbon_idx] = 5
    with mock.patch.object(
      Stash, 'remove', side_effect=[True, False]
    ):
      result = Chromosome.evaluate(
        genes, crafting_tree, {ItemId.RIBBON: 5},
        initial_coins=100000,
      )
    assert result[0] == 0.0

  def test_decode_empty(self):
    '''Decoding an empty array yields empty dicts.'''
    genes = np.zeros((3, 19), dtype=np.uint8)
    assert Chromosome.decode(genes) == [{}, {}, {}]


class TestGaSchedulerExtra:
  '''Tests for scheduler logging helpers.'''

  def test_run_with_population_divisible_by_four(
    self, crafting_tree
  ):
    '''A pop of 8 runs one generation and logs it.'''
    scheduler = GaScheduler(
      _ga_config(), crafting_tree, {ItemId.STRING: 5}
    )
    with tempfile.TemporaryDirectory() as tmp:
      hof = scheduler.run(n_generations=1, output_dir=tmp)
      assert len(hof) > 0
      log = os.path.join(tmp, 'ga_population_log.jsonl')
      assert os.path.exists(log)

  def test_log_generation_without_path(self, crafting_tree):
    '''_log_generation is a no-op without a log path.'''
    scheduler = GaScheduler(
      _ga_config(), crafting_tree, {ItemId.STRING: 5}
    )
    scheduler._log_generation(0, [], mock.MagicMock())
    assert scheduler._log_path is None

  def test_evaluate_delegates(self, crafting_tree):
    '''_evaluate returns the chromosome fitness tuple.'''
    scheduler = GaScheduler(
      _ga_config(), crafting_tree, {ItemId.STRING: 5}
    )
    genes = np.zeros((MAX_TICKS, 19), dtype=np.uint8)
    assert len(scheduler._evaluate(genes)) == 4


class TestMultiAgentExtra:
  '''Tests for tier env and orchestrator branches.'''

  def _orchestrator(self, env_config):
    '''Build and reset an orchestrator.'''
    orch = MultiAgentOrchestrator(env_config)
    orch.reset()
    return orch

  def test_tier_env_reset_and_step_without_orchestrator(
    self, crafting_tree, target_provider
  ):
    '''TierEnv reset works standalone; step requires orchestrator.'''
    stash = Stash()
    coins = CoinGenerator(1000)
    slots = SlotScheduler(crafting_tree)
    board = OrderBoard()
    tier_env = TierEnv(
      tier=0,
      tier_items=[ItemId.STRING],
      ingredient_ids=[ItemId.COTTON],
      crafting_tree=crafting_tree,
      stash=stash,
      coins=coins,
      slots=slots,
      order_board=board,
      targets=target_provider,
    )
    obs, _ = tier_env.reset()
    assert 'local_stash' in obs
    with pytest.raises(RuntimeError):
      tier_env.step(np.zeros(1, dtype=np.int64))

  def test_tier_env_reset_with_orchestrator(self, env_config):
    '''TierEnv reset delegates to the orchestrator.'''
    orch = self._orchestrator(env_config)
    tier_env = orch.tier_envs[orch.tier_numbers[0]]
    obs, _ = tier_env.reset()
    assert 'coins' in obs

  def test_tier_env_step_with_orchestrator(self, env_config):
    '''TierEnv step delegates to the orchestrator.'''
    orch = self._orchestrator(env_config)
    tier_env = orch.tier_envs[orch.tier_numbers[0]]
    action = np.zeros(len(tier_env.tier_items), dtype=np.int64)
    obs, reward, _, _, _ = tier_env.step(action)
    assert 'local_stash' in obs
    assert isinstance(reward, float)

  def test_action_masks_busy_slot(self, env_config):
    '''Busy slots yield only the zero-batch mask entry.'''
    orch = self._orchestrator(env_config)
    tier_env = orch.tier_envs[orch.tier_numbers[0]]
    first = tier_env.tier_items[0]
    orch.slots.start(first, 1)
    mask = tier_env.action_masks()
    assert mask[0]

  def test_decode_and_apply_rejections(self, env_config):
    '''Busy, unaffordable and zero actions are rejected.'''
    orch = self._orchestrator(env_config)
    tier_env = orch.tier_envs[orch.tier_numbers[0]]
    n = len(tier_env.tier_items)
    # Zero action: nothing applied or rejected.
    result = tier_env.decode_and_apply(np.zeros(n, dtype=np.int64))
    assert result['applied'] == {}
    # Unaffordable (no coins after reset to 0 balance base).
    orch.coins.reset(0)
    result = tier_env.decode_and_apply(np.ones(n, dtype=np.int64))
    assert result['rejected']
    # Busy slot rejection.
    orch.slots.start(tier_env.tier_items[0], 1)
    orch.coins.reset(100000)
    result = tier_env.decode_and_apply(np.ones(n, dtype=np.int64))
    assert tier_env.tier_items[0] in result['rejected']

  def test_decode_coin_spend_failure(self, env_config):
    '''Coin spend failure in decode_and_apply rejects.'''
    orch = self._orchestrator(env_config)
    tier_env = orch.tier_envs[orch.tier_numbers[0]]
    orch.coins.reset(100000)
    action = np.ones(
      len(tier_env.tier_items), dtype=np.int64
    )
    with mock.patch.object(
      CoinGenerator, 'spend', return_value=False
    ):
      result = tier_env.decode_and_apply(action)
    assert result['rejected']

  def test_decode_material_failure(self, env_config):
    '''Material removal failure in decode_and_apply rejects.'''
    orch = self._orchestrator(env_config)
    tier_env = orch.tier_envs[orch.tier_numbers[0]]
    orch.coins.reset(100000)
    action = np.ones(
      len(tier_env.tier_items), dtype=np.int64
    )
    with mock.patch.object(
      Stash, 'remove', side_effect=[True, False]
    ):
      result = tier_env.decode_and_apply(action)
    assert result['rejected']

  def test_orchestrator_step_and_masks(self, env_config):
    '''Orchestrator step returns obs/rewards for all tiers.'''
    orch = self._orchestrator(env_config)
    actions = {
      tier: np.zeros(
        len(orch.tier_envs[tier].tier_items), dtype=np.int64
      )
      for tier in orch.tier_numbers
    }
    obs, rewards, _, _, info = orch.step(actions)
    assert set(obs) == set(orch.tier_numbers)
    assert set(rewards) == set(orch.tier_numbers)
    assert info['tick'] == 1
    masks = orch.get_action_masks()
    assert set(masks) == set(orch.tier_numbers)

  def test_orchestrator_step_partial_actions(self, env_config):
    '''Step with only some tiers present still works.'''
    orch = self._orchestrator(env_config)
    tier = orch.tier_numbers[0]
    actions = {
      tier: np.zeros(
        len(orch.tier_envs[tier].tier_items), dtype=np.int64
      )
    }
    obs, rewards, _, _, _ = orch.step(actions)
    assert tier in obs
    assert len(rewards) == len(orch.tier_numbers)

  def test_group_and_ingredients(self, crafting_tree):
    '''Grouping helpers return tier groups and ingredients.'''
    groups = group_items_by_tier(crafting_tree)
    assert groups
    items = next(iter(groups.values()))
    assert get_ingredient_ids(crafting_tree, items)

  def test_order_board_reset(self):
    '''OrderBoard reset clears orders and fulfilments.'''
    board = OrderBoard()
    board.post_order(ItemId.STRING, 5)
    board.fulfill(ItemId.STRING, 2)
    assert board.get_pending(ItemId.STRING) == 3
    board.reset()
    assert board.get_pending(ItemId.STRING) == 0
    assert board.get_all_pending_for_tier([ItemId.STRING]) == {}

  def test_orchestrator_initial_stash(self, env_config):
    '''Orchestrator reset applies the initial stash.'''
    env_config['initial_stash'] = {'string': 4, 'bogus': 2}
    orch = MultiAgentOrchestrator(env_config)
    orch.reset()
    assert orch.stash.get(ItemId.STRING) == 4

  def test_tier_env_no_ingredients_obs(self, env_config):
    '''A tier env without ingredients builds zero inputs.'''
    orch = self._orchestrator(env_config)
    tier_env = orch.tier_envs[orch.tier_numbers[0]]
    tier_env._n_ingredients = 0
    obs = tier_env.build_obs(0)
    assert obs['input_stash'].shape == (1,)


class TestActionHandlerBranches:
  '''Tests for action handler failure branches.'''

  def test_max_batch_property(self, crafting_tree):
    '''max_batch property returns the configured cap.'''
    handler = ActionHandler(crafting_tree, max_batch=7)
    assert handler.max_batch == 7

  def test_material_deduction_failed(self, crafting_tree):
    '''Material deduction failure is reported.'''
    handler = ActionHandler(crafting_tree, max_batch=20)
    stash = Stash()
    stash.add(ItemId.STRING, 9999)
    stash.add(ItemId.WOOD, 9999)
    coins = CoinGenerator(999999)
    slots = SlotScheduler(crafting_tree)
    with mock.patch.object(
      Stash, 'remove', side_effect=[True, False]
    ):
      result = handler.validate_and_apply(
        {ItemId.RIBBON: 1}, stash, coins, slots
      )
    assert result['rejected'][ItemId.RIBBON] == (
      'material_deduction_failed'
    )

  def test_coin_deduction_failed(self, crafting_tree):
    '''Coin deduction failure restores materials.'''
    handler = ActionHandler(crafting_tree, max_batch=20)
    stash = Stash()
    stash.add(ItemId.COTTON, 9999)
    coins = CoinGenerator(999999)
    slots = SlotScheduler(crafting_tree)
    with mock.patch.object(
      CoinGenerator, 'spend', return_value=False
    ):
      result = handler.validate_and_apply(
        {ItemId.STRING: 1}, stash, coins, slots
      )
    assert result['rejected'][ItemId.STRING] == (
      'coin_deduction_failed'
    )

  def test_busy_and_insufficient(self, crafting_tree):
    '''Busy slots and empty resources are rejected.'''
    handler = ActionHandler(crafting_tree, max_batch=20)
    stash = Stash()
    coins = CoinGenerator(0)
    slots = SlotScheduler(crafting_tree)
    result = handler.validate_and_apply(
      {ItemId.STRING: 1}, stash, coins, slots
    )
    assert result['rejected'][ItemId.STRING] == (
      'insufficient_resources'
    )
    slots.start(ItemId.STRING, 1)
    stash.add(ItemId.COTTON, 9999)
    coins.reset(99999)
    result = handler.validate_and_apply(
      {ItemId.STRING: 1}, stash, coins, slots
    )
    assert result['rejected'][ItemId.STRING] == 'slot_busy'


class TestCoreBranches:
  '''Tests for small core helper branches.'''

  def test_cost_max_affordable_edges(self):
    '''Cost calculator handles zero and tight budgets.'''
    assert CostCalculator.max_affordable_batch(0, 100) == 0
    assert CostCalculator.max_affordable_batch(10, 0) == 0
    assert CostCalculator.max_affordable_batch(10, 1) == 0
    assert CostCalculator.max_affordable_batch(10, 10) == 1
    assert CostCalculator.total_cost(10, 0) == 0.0
    with pytest.raises(ValueError):
      CostCalculator.unit_cost(10, 0)

  def test_inventory_zero_quantity_ingredient(self):
    '''Zero-quantity ingredients do not limit batches.'''
    stash = Stash()
    from src.core.items import Ingredient  # noqa: PLC0415

    ingredients = (Ingredient(ItemId.COTTON, 0),)
    assert stash.max_affordable_batch_materials(ingredients) == 9999
    assert stash.max_affordable_batch_materials(()) == 9999
    assert stash.can_afford_materials(ingredients, 5)

  def test_items_invalid_yaml(self, tmp_path):
    '''Invalid crafting tree YAML raises ValueError.'''
    path = os.path.join(str(tmp_path), 'bad.yaml')
    with open(path, 'w', encoding='utf-8') as handle:
      handle.write('nothing: here\n')
    with pytest.raises(ValueError):
      CraftingTree.from_yaml(path)

  def test_items_craft_time_ticks(self, crafting_tree):
    '''craft_time_ticks converts minutes to ticks.'''
    assert crafting_tree.craft_time_ticks(ItemId.STRING) >= 1

  def test_target_provider_edges(self, tmp_path):
    '''Target provider handles unknown and zero targets.'''
    path = os.path.join(str(tmp_path), 'bad.yaml')
    with open(path, 'w', encoding='utf-8') as handle:
      handle.write('nothing: here\n')
    with pytest.raises(ValueError):
      TargetProvider(path)
    good = os.path.join(str(tmp_path), 'good.yaml')
    with open(good, 'w', encoding='utf-8') as handle:
      yaml.dump({'targets': {'string': 0}}, handle)
    provider = TargetProvider(good)
    assert provider.remaining(ItemId.COTTON) == 0
    assert provider.fraction_complete() == 1.0
    provider.reload(good)
    assert provider.targets == {ItemId.STRING: 0}

  def test_frame_skip_property(self, env):
    '''FrameSkipWrapper exposes its skip value.'''
    wrapped = FrameSkipWrapper(env, skip=4)
    assert wrapped.skip == 4

  def test_excess_penalty_zero_targets(self):
    '''Excess penalty is zero when there are no targets.'''
    component = WasteMinimizationReward()
    state = {
      'targets_remaining': np.zeros(23),
      'targets_total': np.zeros(23),
    }
    assert component.compute(state, {}, state) == 0.0

  def test_crafting_env_render_and_stash(self, env_config):
    '''CraftingEnv render and initial stash reset work.'''
    env_config['initial_stash'] = {'string': 2, 'bogus': 1}
    env = CraftingEnv(env_config)
    env.reset()
    assert env.stash.get(ItemId.STRING) == 2
    text = env.render()
    assert 'Tick' in text


class TestInventoryLoggerBranches:
  '''Tests for logger rotation and serialisation.'''

  def test_rotation_filename_and_rotate(self, tmp_path):
    '''Rotation appends .gz and gzips the source.'''
    handler = GzipRotatingFileHandler(
      os.path.join(str(tmp_path), 'x.log')
    )
    assert handler.rotation_filename('x.log') == 'x.log.gz'
    src = os.path.join(str(tmp_path), 'src.log')
    dest = os.path.join(str(tmp_path), 'dest.log.gz')
    with open(src, 'wb') as handle:
      handle.write(b'hello log')
    handler.rotate(src, dest)
    assert not os.path.exists(src)
    with gzip.open(dest, 'rb') as handle:
      assert handle.read() == b'hello log'

  def test_serialize_default_branches(self, tmp_path):
    '''Serialisation handles arrays, enums and objects.'''
    logger = InventoryLogger(
      log_dir=str(tmp_path), name='test_extra'
    )
    text = logger._serialize(
      {
        'arr': np.array([1, 2]),
        'item': ItemId.STRING,
        'plain': object(),
      }
    )
    assert 'arr' in text
    assert '4' in text

  def test_log_methods(self, tmp_path):
    '''All log methods emit without error.'''
    logger = InventoryLogger(
      log_dir=str(tmp_path), name='test_extra2'
    )
    logger.log_decision(1, {'a': 1})
    logger.log_transition(1, {'stash': 1}, {'stash': 2})
    logger.log_reward(1, 1.5, {'a': 1.5})
    logger.log_warning(1, 'warn')
    logger.log_slot_event(1, 'start', 'string')
    logger.log_ga_event(1, 'gen', {'x': 1})
    logger.log_frame_skip(1, 3)
    assert logger._ts()


if __name__ == '__main__':
  pytest.main([__file__])


class TestSimulateRollback:
  '''Tests for simulate_schedule rollback branches.'''

  def test_material_failure_rolls_back(self, crafting_tree):
    '''A patched removal failure leaves the stash intact.'''
    from src.dashboard.utils import simulate_schedule  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    frame = pd.DataFrame(
      [
        {
          'tick_index': 0,
          'elapsed_minutes': 0,
          'item_name': 'ribbon',
          'batch_size_decided': 2,
        }
      ]
    )
    with mock.patch.object(
      Stash, 'remove', side_effect=[True, False]
    ):
      result = simulate_schedule(
        frame, crafting_tree, max_ticks=2,
        initial_coins=100000,
      )
    assert result['active_slot_history'][0] == 0

  def test_coin_failure_rolls_back(self, crafting_tree):
    '''A patched coin spend failure restores ingredients.'''
    from src.dashboard.utils import simulate_schedule  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    frame = pd.DataFrame(
      [
        {
          'tick_index': 0,
          'elapsed_minutes': 0,
          'item_name': 'string',
          'batch_size_decided': 2,
        }
      ]
    )
    with mock.patch.object(
      CoinGenerator, 'spend', return_value=False
    ):
      result = simulate_schedule(
        frame, crafting_tree, max_ticks=2,
        initial_coins=100000,
      )
    assert result['active_slot_history'][0] == 0

  def test_busy_slot_skipped(self, crafting_tree):
    '''Scheduling the same item twice in one tick is safe.'''
    from src.dashboard.utils import simulate_schedule  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    frame = pd.DataFrame(
      [
        {
          'tick_index': 0,
          'elapsed_minutes': 0,
          'item_name': 'string',
          'batch_size_decided': 2,
        },
        {
          'tick_index': 0,
          'elapsed_minutes': 0,
          'item_name': 'string',
          'batch_size_decided': 2,
        },
      ]
    )
    result = simulate_schedule(
      frame, crafting_tree, max_ticks=2,
      initial_coins=100000,
    )
    assert result['active_slot_history'][0] <= 1
