from __future__ import annotations

from typing import Dict

import numpy as np
import yaml

from src.core.items import NUM_ITEMS, ItemId, ITEM_NAME_TO_ID


class TargetProvider:
    def __init__(self, targets_path: str):
        self._targets_path = targets_path
        self._targets: Dict[ItemId, int] = {}
        self._load(targets_path)
        self._delivered: Dict[ItemId, int] = {k: 0 for k in self._targets}

    def _load(self, path: str) -> None:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self._targets = {}
        for name, count in data["targets"].items():
            item_id = ITEM_NAME_TO_ID[name.lower()]
            self._targets[item_id] = count

    def reload(self, path: str | None = None) -> None:
        self._load(path or self._targets_path)
        self._delivered = {k: 0 for k in self._targets}

    @property
    def targets(self) -> Dict[ItemId, int]:
        return dict(self._targets)

    def deliver(self, item_id: ItemId, qty: int) -> None:
        if item_id in self._targets:
            self._delivered[item_id] = self._delivered.get(item_id, 0) + qty

    def remaining(self, item_id: ItemId) -> int:
        if item_id not in self._targets:
            return 0
        return max(0, self._targets[item_id] - self._delivered.get(item_id, 0))

    def is_complete(self) -> bool:
        return all(
            self._delivered.get(k, 0) >= v for k, v in self._targets.items()
        )

    def targets_remaining_array(self) -> np.ndarray:
        arr = np.zeros(NUM_ITEMS, dtype=np.int32)
        for item_id, target in self._targets.items():
            delivered = self._delivered.get(item_id, 0)
            arr[int(item_id)] = target - delivered
        return arr

    def targets_total_array(self) -> np.ndarray:
        arr = np.zeros(NUM_ITEMS, dtype=np.int32)
        for item_id, target in self._targets.items():
            arr[int(item_id)] = target
        return arr

    def fraction_complete(self) -> float:
        total_required = sum(self._targets.values())
        if total_required == 0:
            return 1.0
        total_delivered = sum(
            min(self._delivered.get(k, 0), v) for k, v in self._targets.items()
        )
        return total_delivered / total_required

    def reset(self) -> None:
        self._delivered = {k: 0 for k in self._targets}
