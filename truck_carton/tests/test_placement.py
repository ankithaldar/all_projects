from truck_carton.domain.models import (
    Carton,
    PlacementInfo,
)
from truck_carton.packing.placement import (
    PlacementValidator,
)
from truck_carton.packing.rotation import Rotation
from truck_carton.packing.space3d import Space3D


def test_check_bounds_valid():
    space = Space3D(8, 4, 4)
    v = PlacementValidator(space, {})
    assert v.check_bounds(
        0, 0, 0, 2, 2, 2
    ) is True


def test_check_bounds_invalid():
    space = Space3D(8, 4, 4)
    v = PlacementValidator(space, {})
    assert v.check_bounds(
        7, 3, 3, 2, 2, 2
    ) is False


def test_check_gravity_floor():
    space = Space3D(8, 4, 4)
    v = PlacementValidator(space, {})
    assert v.check_gravity_support(
        0, 0, 0, 2, 2, 2
    ) is True


def test_check_gravity_unsupported():
    space = Space3D(8, 4, 4)
    v = PlacementValidator(space, {})
    assert v.check_gravity_support(
        0, 0, 1, 2, 2, 2
    ) is False


def test_check_gravity_supported():
    space = Space3D(8, 4, 4)
    space.place(1, 0, 0, 0, 2, 2, 1)
    v = PlacementValidator(space, {})
    assert v.check_gravity_support(
        0, 0, 1, 2, 2, 1
    ) is True


def test_check_weight_within():
    space = Space3D(8, 4, 4)
    v = PlacementValidator(space, {})
    assert v.check_weight(
        10.0, 90.0, 100.0
    ) is True


def test_check_weight_exceeded():
    space = Space3D(8, 4, 4)
    v = PlacementValidator(space, {})
    assert v.check_weight(
        11.0, 90.0, 100.0
    ) is False


def test_validate_placement_valid():
    space = Space3D(8, 4, 4)
    v = PlacementValidator(space, {})
    carton = Carton(
        1, 2, 2, 2, 10.0, False, 2, 0
    )
    valid, violations = v.validate_placement(
        carton, 0, 0, 0, 2, 2, 2, 0.0, 500.0
    )
    assert valid is True
    assert violations == []


def test_validate_placement_overlap():
    space = Space3D(8, 4, 4)
    space.place(1, 0, 0, 0, 2, 2, 2)
    v = PlacementValidator(space, {})
    carton = Carton(
        2, 2, 2, 2, 10.0, False, 2, 0
    )
    valid, violations = v.validate_placement(
        carton, 0, 0, 0, 2, 2, 2, 0.0, 500.0
    )
    assert valid is False
    assert 'overlap' in violations


def test_find_valid_positions_empty_space():
    space = Space3D(4, 4, 4)
    v = PlacementValidator(space, {})
    carton = Carton(
        1, 1, 1, 1, 5.0, False, 2, 0
    )
    positions = v.find_valid_positions(
        carton, 0.0, 500.0
    )
    assert len(positions) > 0
    for x, y, z, rot in positions:
        assert z == 0


def test_find_valid_positions_with_existing():
    space = Space3D(4, 4, 4)
    space.place(1, 0, 0, 0, 2, 2, 1)
    placed = {
        1: PlacementInfo(
            0, (0, 0, 0), (2, 2, 1), Rotation.LWH
        )
    }
    v = PlacementValidator(space, placed)
    carton = Carton(
        2, 2, 2, 1, 5.0, False, 2, 0
    )
    positions = v.find_valid_positions(
        carton, 5.0, 500.0
    )
    has_stacked = any(
        z == 1 for _, _, z, _ in positions
    )
    assert has_stacked


def test_find_valid_positions_weight_exceeded():
    space = Space3D(4, 4, 4)
    v = PlacementValidator(space, {})
    carton = Carton(
        1, 1, 1, 1, 50.0, False, 2, 0
    )
    positions = v.find_valid_positions(
        carton, 460.0, 500.0
    )
    assert len(positions) == 0


def test_fragile_rotations_only():
    space = Space3D(4, 4, 4)
    v = PlacementValidator(space, {})
    carton = Carton(
        1, 2, 3, 4, 5.0, True, 2, 0
    )
    positions = v.find_valid_positions(
        carton, 0.0, 500.0
    )
    for _, _, _, rot in positions:
        assert rot in (
            Rotation.LWH, Rotation.WLH
        )


def test_validate_placement_fragility_with_lookup():
    """validate_placement must detect fragility
    violations when all_cartons is provided."""
    space = Space3D(4, 4, 4)
    fragile = Carton(
        1, 2, 2, 1, 5.0, True, 3, 0
    )
    space.place(1, 0, 0, 0, 2, 2, 1)
    placed = {
        1: PlacementInfo(
            0, (0, 0, 0), (2, 2, 1), Rotation.LWH
        )
    }
    v = PlacementValidator(space, placed)
    non_fragile = Carton(
        2, 2, 2, 1, 10.0, False, 1, 0
    )
    valid, violations = v.validate_placement(
        non_fragile, 0, 0, 1, 2, 2, 1,
        5.0, 500.0,
        all_cartons=[fragile, non_fragile],
    )
    assert valid is False
    assert 'fragility_violation' in violations


def test_validate_placement_fragility_no_lookup():
    """Without all_cartons, fragility check is
    incomplete — should still not crash."""
    space = Space3D(4, 4, 4)
    fragile = Carton(
        1, 2, 2, 1, 5.0, True, 3, 0
    )
    space.place(1, 0, 0, 0, 2, 2, 1)
    placed = {
        1: PlacementInfo(
            0, (0, 0, 0), (2, 2, 1), Rotation.LWH
        )
    }
    v = PlacementValidator(space, placed)
    non_fragile = Carton(
        2, 2, 2, 1, 10.0, False, 1, 0
    )
    # Without all_cartons, lookup is empty, so
    # fragility check passes (incomplete but safe).
    valid, _ = v.validate_placement(
        non_fragile, 0, 0, 1, 2, 2, 1,
        5.0, 500.0,
    )
    assert isinstance(valid, bool)


def test_fragile_above_fragile_allowed():
    """Fragile cartons should be placeable on top
    of other fragile cartons."""
    space = Space3D(4, 4, 4)
    f1 = Carton(1, 2, 2, 1, 3.0, True, 2, 0)
    f2 = Carton(2, 2, 2, 1, 3.0, True, 2, 0)
    space.place(1, 0, 0, 0, 2, 2, 1)
    placed = {
        1: PlacementInfo(
            0, (0, 0, 0), (2, 2, 1), Rotation.LWH
        )
    }
    v = PlacementValidator(space, placed)
    positions = v.find_valid_positions(
        f2, 3.0, 500.0,
        all_cartons=[f1, f2],
    )
    stacked = [
        (x, y, z, r) for x, y, z, r in positions
        if z == 1 and x == 0 and y == 0
    ]
    assert len(stacked) > 0


def test_nonfragile_above_fragile_rejected():
    """Non-fragile cartons must not be placed above
    fragile cartons."""
    space = Space3D(4, 4, 4)
    fragile = Carton(
        1, 2, 2, 1, 3.0, True, 2, 0
    )
    non_fragile = Carton(
        2, 2, 2, 1, 5.0, False, 1, 0
    )
    space.place(1, 0, 0, 0, 2, 2, 1)
    placed = {
        1: PlacementInfo(
            0, (0, 0, 0), (2, 2, 1), Rotation.LWH
        )
    }
    v = PlacementValidator(space, placed)
    positions = v.find_valid_positions(
        non_fragile, 3.0, 500.0,
        all_cartons=[fragile, non_fragile],
    )
    # No position at (0, 0, 1) should exist since
    # fragile carton is below.
    stacked_on_fragile = [
        (x, y, z, r) for x, y, z, r in positions
        if z == 1 and x == 0 and y == 0
    ]
    assert len(stacked_on_fragile) == 0
