#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for items and the crafting tree.'''

from __future__ import annotations

import pytest
import yaml

from src.core.items import (
  ItemId, CraftingTree, Ingredient, Recipe,
  NUM_ITEMS, NUM_CRAFTABLE, BASE_ITEM_IDS, CRAFTABLE_ITEM_IDS,
  ITEM_NAME_TO_ID, MAX_AFFORDABLE_BATCH,
)


def _write_tree_yaml(tmp_path, craft_time=5, quantity=3,
                     item_id='string', ingredient_item='cotton'):
  data = {
    'items': [
      {'id': 'cotton', 'type': 'base', 'tier': 0},
      {
        'id': item_id,
        'type': 'craftable',
        'tier': 1,
        'recipe': {
          'ingredients': [
            {'item': ingredient_item, 'quantity': quantity},
          ],
          'coin_cost': 50,
          'craft_time': craft_time,
        },
      },
    ]
  }
  path = tmp_path / 'tree.yaml'
  path.write_text(yaml.safe_dump(data), encoding='utf-8')
  return str(path)


class TestItemId:
  '''Tests for item identifiers.'''
  def test_num_items(self):
    assert len(ItemId) == NUM_ITEMS == 23

  def test_base_items(self):
    assert ItemId.COTTON == 0
    assert ItemId.TREE == 1
    assert ItemId.ROCK == 2
    assert ItemId.QUARTZ == 3

  def test_craftable_count(self):
    assert NUM_CRAFTABLE == 19
    assert len(CRAFTABLE_ITEM_IDS) == 19

  def test_base_item_ids(self):
    assert len(BASE_ITEM_IDS) == 4

  def test_craftable_item_ids_span(self):
    assert CRAFTABLE_ITEM_IDS[0] == ItemId.STRING
    assert CRAFTABLE_ITEM_IDS[-1] == ItemId.ARTIFACT

  def test_name_to_id_mapping(self):
    assert ITEM_NAME_TO_ID['cotton'] == ItemId.COTTON
    assert ITEM_NAME_TO_ID['artifact'] == ItemId.ARTIFACT
    assert len(ITEM_NAME_TO_ID) == NUM_ITEMS


class TestCraftingTree:
  '''Tests for the crafting tree.'''
  def test_load_from_yaml(self, crafting_tree: CraftingTree):
    assert len(crafting_tree.recipes) == NUM_CRAFTABLE

  def test_all_craftable_have_recipes(self, crafting_tree: CraftingTree):
    for item_id_int in CRAFTABLE_ITEM_IDS:
      assert ItemId(item_id_int) in crafting_tree.recipes

  def test_topo_order_length(self, crafting_tree: CraftingTree):
    assert len(crafting_tree.topo_order) == NUM_CRAFTABLE

  def test_topo_order_respects_deps(self, crafting_tree: CraftingTree):
    order_idx = {
      item_id: i for i, item_id in enumerate(crafting_tree.topo_order)
    }
    for item_id, recipe in crafting_tree.recipes.items():
      for ing in recipe.ingredients:
        if ing.item_id in order_idx:
          assert order_idx[ing.item_id] < order_idx[item_id], (
            f'{ing.item_id.name} should come before {item_id.name}'
          )

  def test_tier_assignments(self, crafting_tree: CraftingTree):
    assert crafting_tree.tier[ItemId.COTTON] == 0
    assert crafting_tree.tier[ItemId.STRING] == 1
    assert crafting_tree.tier[ItemId.RIBBON] == 2
    assert crafting_tree.tier[ItemId.ARTIFACT] == 8

  def test_string_recipe(self, crafting_tree: CraftingTree):
    recipe = crafting_tree.get_recipe(ItemId.STRING)
    assert recipe.coin_cost == 50
    assert recipe.craft_time == 5
    assert len(recipe.ingredients) == 1
    assert recipe.ingredients[0].item_id == ItemId.COTTON
    assert recipe.ingredients[0].quantity == 3

  def test_artifact_recipe(self, crafting_tree: CraftingTree):
    recipe = crafting_tree.get_recipe(ItemId.ARTIFACT)
    assert recipe.coin_cost == 10000
    assert recipe.craft_time == 4320
    ing_ids = {ing.item_id for ing in recipe.ingredients}
    assert ing_ids == {ItemId.NECKLACE, ItemId.ELEMENTSTONE}

  def test_craft_time_ticks(self, crafting_tree: CraftingTree):
    assert crafting_tree.craft_time_ticks(ItemId.STRING) == 1
    assert crafting_tree.craft_time_ticks(ItemId.METAL) == 3
    assert crafting_tree.craft_time_ticks(ItemId.ARTIFACT) == 864

  def test_is_base(self, crafting_tree: CraftingTree):
    assert crafting_tree.is_base(ItemId.COTTON)
    assert not crafting_tree.is_base(ItemId.STRING)

  def test_ribbon_recipe(self, crafting_tree: CraftingTree):
    recipe = crafting_tree.get_recipe(ItemId.RIBBON)
    assert recipe.coin_cost == 100
    assert recipe.craft_time == 15
    ing_ids = {ing.item_id for ing in recipe.ingredients}
    assert ing_ids == {ItemId.STRING, ItemId.WOOD}


class TestCraftingTreeValidation:
  '''Tests for F2/F4/F5 validation in from_yaml.'''

  @pytest.mark.parametrize('craft_time', [-5, 0])
  def test_from_yaml_nonpositive_craft_time_raises(
    self, tmp_path, craft_time
  ):
    path = _write_tree_yaml(tmp_path, craft_time=craft_time)
    with pytest.raises(ValueError, match='craft_time'):
      CraftingTree.from_yaml(path)

  @pytest.mark.parametrize('quantity', [-1, 0])
  def test_from_yaml_nonpositive_ingredient_quantity_raises(
    self, tmp_path, quantity
  ):
    path = _write_tree_yaml(tmp_path, quantity=quantity)
    with pytest.raises(ValueError, match='quantity'):
      CraftingTree.from_yaml(path)

  def test_from_yaml_unknown_item_id_raises_with_name(self, tmp_path):
    path = _write_tree_yaml(tmp_path, item_id='no_such_item')
    with pytest.raises(ValueError, match='no_such_item'):
      CraftingTree.from_yaml(path)

  def test_from_yaml_unknown_ingredient_raises_with_name(self, tmp_path):
    path = _write_tree_yaml(tmp_path, ingredient_item='no_such_thing')
    with pytest.raises(ValueError, match='no_such_thing'):
      CraftingTree.from_yaml(path)

  @pytest.mark.parametrize('craft_time', [-5, 0])
  def test_craft_time_ticks_nonpositive_raises(self, craft_time):
    recipe = Recipe(
      output_id=ItemId.STRING,
      ingredients=(Ingredient(ItemId.COTTON, 3),),
      coin_cost=50,
      craft_time=craft_time,
    )
    tree = CraftingTree(
      recipes={ItemId.STRING: recipe},
      topo_order=[ItemId.STRING],
      tier={ItemId.STRING: 1},
    )
    with pytest.raises(ValueError, match='craft_time'):
      tree.craft_time_ticks(ItemId.STRING)

  def test_max_affordable_batch_constant(self):
    assert MAX_AFFORDABLE_BATCH == 9999
