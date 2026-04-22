from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

import yaml


class ItemId(IntEnum):
    COTTON = 0
    TREE = 1
    ROCK = 2
    QUARTZ = 3
    STRING = 4
    WOOD = 5
    METAL = 6
    BRONZE = 7
    AMETHYST = 8
    ORB = 9
    RIBBON = 10
    NEEDLES = 11
    SPARKLES = 12
    SILVER = 13
    GOLD = 14
    PENDANT = 15
    WATER = 16
    NECKLACE = 17
    FIRE = 18
    WATERSTONE = 19
    FIRESTONE = 20
    ELEMENTSTONE = 21
    ARTIFACT = 22


NUM_ITEMS = 23
NUM_CRAFTABLE = 19
BASE_ITEM_IDS = [ItemId.COTTON, ItemId.TREE, ItemId.ROCK, ItemId.QUARTZ]
CRAFTABLE_ITEM_IDS = list(range(4, 23))

ITEM_NAME_TO_ID: Dict[str, ItemId] = {item.name.lower(): item for item in ItemId}


@dataclass(frozen=True)
class Ingredient:
    item_id: ItemId
    quantity: int


@dataclass(frozen=True)
class Recipe:
    output_id: ItemId
    ingredients: Tuple[Ingredient, ...]
    coin_cost: int
    craft_time: int  # minutes


class CraftingTree:
    def __init__(
        self,
        recipes: Dict[ItemId, Recipe],
        topo_order: List[ItemId],
        tier: Dict[ItemId, int],
    ):
        self.recipes = recipes
        self.topo_order = topo_order
        self.tier = tier

    @classmethod
    def from_yaml(cls, path: str) -> CraftingTree:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None or "items" not in data:
            raise ValueError(f"Invalid crafting tree YAML: {path}")

        recipes: Dict[ItemId, Recipe] = {}
        tier_map: Dict[ItemId, int] = {}

        for entry in data["items"]:
            item_id = ITEM_NAME_TO_ID[entry["id"]]
            tier_map[item_id] = entry["tier"]

            if entry["type"] == "craftable":
                recipe_data = entry["recipe"]
                ingredients = tuple(
                    Ingredient(
                        item_id=ITEM_NAME_TO_ID[ing["item"]],
                        quantity=ing["quantity"],
                    )
                    for ing in recipe_data["ingredients"]
                )
                recipes[item_id] = Recipe(
                    output_id=item_id,
                    ingredients=ingredients,
                    coin_cost=recipe_data["coin_cost"],
                    craft_time=recipe_data["craft_time"],
                )

        topo_order = cls._topological_sort(recipes)
        return cls(recipes=recipes, topo_order=topo_order, tier=tier_map)

    @staticmethod
    def _topological_sort(recipes: Dict[ItemId, Recipe]) -> List[ItemId]:
        in_degree: Dict[ItemId, int] = {item_id: 0 for item_id in recipes}
        dependents: Dict[ItemId, List[ItemId]] = {item_id: [] for item_id in recipes}

        for item_id, recipe in recipes.items():
            for ing in recipe.ingredients:
                if ing.item_id in recipes:
                    in_degree[item_id] += 1
                    dependents[ing.item_id].append(item_id)

        queue = [item_id for item_id, deg in in_degree.items() if deg == 0]
        queue.sort(key=lambda x: int(x))
        result: List[ItemId] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for dep in sorted(dependents[node], key=lambda x: int(x)):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
                    queue.sort(key=lambda x: int(x))

        if len(result) != len(recipes):
            raise ValueError("Cyclic dependency detected in crafting tree")

        return result

    def craft_time_ticks(self, item_id: ItemId) -> int:
        return math.ceil(self.recipes[item_id].craft_time / 5)

    def get_recipe(self, item_id: ItemId) -> Recipe:
        return self.recipes[item_id]

    def is_base(self, item_id: ItemId) -> bool:
        return item_id in BASE_ITEM_IDS
