from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from src.core.items import CraftingTree, ItemId, Ingredient, Recipe, ITEM_NAME_TO_ID
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.core.target_provider import TargetProvider
from src.cat_game_env.crafting_env import CraftingEnv

TREE_PATH = os.path.join(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
  "config", "crafting_tree.yaml",
)
TARGETS_PATH = os.path.join(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
  "config", "targets.yaml",
)


@pytest.fixture
def crafting_tree() -> CraftingTree:
  return CraftingTree.from_yaml(TREE_PATH)


@pytest.fixture
def fresh_stash() -> Stash:
  return Stash()


@pytest.fixture
def funded_coins() -> CoinGenerator:
  return CoinGenerator(100_000)


@pytest.fixture
def slot_scheduler(crafting_tree: CraftingTree) -> SlotScheduler:
  return SlotScheduler(crafting_tree)


@pytest.fixture
def target_provider() -> TargetProvider:
  return TargetProvider(TARGETS_PATH)


@pytest.fixture
def env_config() -> dict:
  return {
    "crafting_tree_path": TREE_PATH,
    "targets_path": TARGETS_PATH,
    "max_batch_size": 20,
    "max_ticks": 2016,
    "initial_coins": 0,
  }


@pytest.fixture
def env(env_config: dict) -> CraftingEnv:
  return CraftingEnv(env_config)


@pytest.fixture
def simple_targets_path() -> str:
  data = {"targets": {"string": 5}}
  with tempfile.NamedTemporaryFile(
    mode="w", suffix=".yaml", delete=False
  ) as f:
    yaml.dump(data, f)
    path = f.name
  yield path
  os.unlink(path)
