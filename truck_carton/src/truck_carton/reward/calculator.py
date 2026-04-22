from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from truck_carton.config import RewardWeights
from truck_carton.domain.models import (
    Carton,
    PlacementInfo,
    Store,
    Truck,
)
from truck_carton.packing.space3d import Space3D


@dataclass
class EnvironmentState:
    """Snapshot passed to each reward component."""

    trucks: list[Truck]
    spaces: list[Space3D]
    placed_cartons: dict[int, PlacementInfo]
    unplaced_cartons: list[Carton]
    all_cartons: list[Carton]
    stores: list[Store]
    current_weights: list[float]
    step_carton: Carton | None
    step_placement: PlacementInfo | None
    is_terminal: bool
    total_cartons: int


class RewardComponent(Protocol):
    def compute(
        self, state: EnvironmentState
    ) -> float: ...


class RewardCalculator:
    """Orchestrates all reward components into a
    weighted sum."""

    def __init__(self, weights: RewardWeights) -> None:
        from truck_carton.reward.completion import (
            CompletionReward,
        )
        from truck_carton.reward.displacement import (
            DisplacementReward,
        )
        from truck_carton.reward.fragility import (
            FragilityReward,
        )
        from truck_carton.reward.grouping import (
            GroupingReward,
        )
        from truck_carton.reward.priority import (
            PriorityReward,
        )
        from truck_carton.reward.support import (
            SupportReward,
        )
        from truck_carton.reward.utilization import (
            UtilizationReward,
        )
        from truck_carton.reward.weight import (
            WeightReward,
        )

        self._components: list[
            tuple[str, float, RewardComponent]
        ] = [
            (
                'utilization',
                weights.alpha_utilization,
                UtilizationReward(),
            ),
            (
                'displacement',
                weights.beta_displacement,
                DisplacementReward(),
            ),
            (
                'grouping',
                weights.gamma_grouping,
                GroupingReward(),
            ),
            (
                'fragility',
                weights.delta_fragility,
                FragilityReward(),
            ),
            (
                'support',
                weights.epsilon_support,
                SupportReward(),
            ),
            (
                'weight',
                weights.zeta_weight,
                WeightReward(),
            ),
            (
                'completion',
                weights.eta_completion,
                CompletionReward(),
            ),
            (
                'priority',
                weights.theta_priority,
                PriorityReward(),
            ),
        ]

    def compute(
        self, state: EnvironmentState
    ) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {}
        total = 0.0
        for name, weight, component in self._components:
            raw = component.compute(state)
            breakdown[name] = raw
            total += weight * raw
        return total, breakdown
