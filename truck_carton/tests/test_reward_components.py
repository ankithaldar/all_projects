from truck_carton.config import RewardWeights
from truck_carton.domain.models import (
    Carton,
    PlacementInfo,
    Store,
    Truck,
)
from truck_carton.packing.rotation import Rotation
from truck_carton.packing.space3d import Space3D
from truck_carton.reward.calculator import (
    EnvironmentState,
    RewardCalculator,
)
from truck_carton.reward.completion import (
    CompletionReward,
)
from truck_carton.reward.grouping import (
    GroupingReward,
)
from truck_carton.reward.support import SupportReward
from truck_carton.reward.utilization import (
    UtilizationReward,
)
from truck_carton.reward.weight import WeightReward


def _make_state(
    spaces=None,
    placed=None,
    cartons=None,
    trucks=None,
    stores=None,
    weights=None,
    total=4,
) -> EnvironmentState:
    trucks = trucks or [
        Truck(0, 4, 4, 4, 500.0, [0, 1])
    ]
    stores = stores or [
        Store(0, 0), Store(1, 1)
    ]
    cartons = cartons or []
    spaces = spaces or [Space3D(4, 4, 4)]
    placed = placed or {}
    weights = weights or [0.0]

    return EnvironmentState(
        trucks=trucks,
        spaces=spaces,
        placed_cartons=placed,
        unplaced_cartons=[],
        all_cartons=cartons,
        stores=stores,
        current_weights=weights,
        step_carton=None,
        step_placement=None,
        is_terminal=False,
        total_cartons=total,
    )


def test_utilization_empty():
    state = _make_state()
    r = UtilizationReward().compute(state)
    assert r == 0.0


def test_utilization_half_filled():
    space = Space3D(4, 4, 4)
    space.place(1, 0, 0, 0, 4, 4, 2)
    state = _make_state(
        spaces=[space], weights=[250.0]
    )
    r = UtilizationReward().compute(state)
    assert r > 0.0


def test_completion_none():
    state = _make_state(total=10)
    r = CompletionReward().compute(state)
    assert r == 0.0


def test_completion_all_placed():
    cartons = [
        Carton(i, 1, 1, 1, 1.0, False, 1, 0)
        for i in range(4)
    ]
    placed = {
        i: PlacementInfo(
            0, (i, 0, 0), (1, 1, 1), Rotation.LWH
        )
        for i in range(4)
    }
    state = _make_state(
        cartons=cartons, placed=placed, total=4
    )
    r = CompletionReward().compute(state)
    assert r == 2.0


def test_completion_partial():
    cartons = [
        Carton(i, 1, 1, 1, 1.0, False, 1, 0)
        for i in range(4)
    ]
    placed = {
        0: PlacementInfo(
            0, (0, 0, 0), (1, 1, 1), Rotation.LWH
        ),
        1: PlacementInfo(
            0, (1, 0, 0), (1, 1, 1), Rotation.LWH
        ),
    }
    state = _make_state(
        cartons=cartons, placed=placed, total=4
    )
    r = CompletionReward().compute(state)
    assert abs(r - 0.5) < 1e-9


def test_weight_no_violation():
    state = _make_state(weights=[400.0])
    r = WeightReward().compute(state)
    assert r == 0.0


def test_weight_violation():
    state = _make_state(weights=[600.0])
    r = WeightReward().compute(state)
    assert r > 0.0


def test_grouping_single_carton():
    cartons = [
        Carton(1, 1, 1, 1, 1.0, False, 1, 0)
    ]
    placed = {
        1: PlacementInfo(
            0, (0, 0, 0), (1, 1, 1), Rotation.LWH
        )
    }
    state = _make_state(
        cartons=cartons, placed=placed
    )
    r = GroupingReward().compute(state)
    assert r == 1.0


def test_grouping_tight():
    cartons = [
        Carton(1, 1, 1, 1, 1.0, False, 1, 0),
        Carton(2, 1, 1, 1, 1.0, False, 1, 0),
    ]
    placed = {
        1: PlacementInfo(
            0, (0, 0, 0), (1, 1, 1), Rotation.LWH
        ),
        2: PlacementInfo(
            0, (1, 0, 0), (1, 1, 1), Rotation.LWH
        ),
    }
    state = _make_state(
        cartons=cartons, placed=placed
    )
    r = GroupingReward().compute(state)
    assert r == 1.0


def test_support_no_violation():
    space = Space3D(4, 4, 4)
    space.place(1, 0, 0, 0, 2, 2, 1)
    space.place(2, 0, 0, 1, 2, 2, 1)
    placed = {
        1: PlacementInfo(
            0, (0, 0, 0), (2, 2, 1), Rotation.LWH
        ),
        2: PlacementInfo(
            0, (0, 0, 1), (2, 2, 1), Rotation.LWH
        ),
    }
    cartons = [
        Carton(1, 2, 2, 1, 1.0, False, 1, 0),
        Carton(2, 2, 2, 1, 1.0, False, 1, 0),
    ]
    state = _make_state(
        spaces=[space], cartons=cartons,
        placed=placed,
    )
    r = SupportReward().compute(state)
    assert r == 0.0


def test_completion_bounded():
    """Completion reward must stay in [0, 1]."""
    cartons = [
        Carton(i, 1, 1, 1, 1.0, False, 1, 0)
        for i in range(4)
    ]
    placed = {
        i: PlacementInfo(
            0, (i, 0, 0), (1, 1, 1), Rotation.LWH
        )
        for i in range(4)
    }
    state = _make_state(
        cartons=cartons, placed=placed, total=4
    )
    r = CompletionReward().compute(state)
    assert 0.0 <= r <= 1.0
    assert r == 1.0


def test_displacement_ignores_off_route_cartons():
    """Cartons destined for stores not on the
    truck's route should not inflate displacement."""
    from truck_carton.reward.displacement import (
        DisplacementReward,
    )
    trucks = [
        Truck(0, 8, 4, 4, 500.0, [0])
    ]
    stores = [Store(0, 0), Store(1, 1)]
    c0 = Carton(1, 1, 1, 1, 1.0, False, 1, 0)
    c1 = Carton(2, 1, 1, 1, 1.0, False, 1, 1)
    placed = {
        1: PlacementInfo(
            0, (0, 0, 0), (1, 1, 1), Rotation.LWH
        ),
        2: PlacementInfo(
            0, (1, 0, 0), (1, 1, 1), Rotation.LWH
        ),
    }
    state = _make_state(
        trucks=trucks, stores=stores,
        cartons=[c0, c1], placed=placed,
        weights=[2.0], total=2,
    )
    r = DisplacementReward().compute(state)
    assert 0.0 <= r <= 1.0


def test_priority_bounded():
    """Priority reward must be in [0, 1]."""
    from truck_carton.reward.priority import (
        PriorityReward,
    )
    cartons = [
        Carton(1, 1, 1, 1, 1.0, False, 3, 0),
        Carton(2, 1, 1, 1, 1.0, False, 1, 0),
    ]
    placed = {
        1: PlacementInfo(
            0, (3, 0, 0), (1, 1, 1), Rotation.LWH
        ),
        2: PlacementInfo(
            0, (0, 0, 0), (1, 1, 1), Rotation.LWH
        ),
    }
    state = _make_state(
        cartons=cartons, placed=placed, total=2
    )
    r = PriorityReward().compute(state)
    assert 0.0 <= r <= 1.0


def test_calculator_returns_breakdown():
    weights = RewardWeights()
    calc = RewardCalculator(weights)
    state = _make_state(total=0)
    total, breakdown = calc.compute(state)
    assert isinstance(total, float)
    assert 'utilization' in breakdown
    assert 'completion' in breakdown
    assert len(breakdown) == 8
