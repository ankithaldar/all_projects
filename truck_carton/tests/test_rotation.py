from truck_carton.packing.rotation import (
    ALL_ROTATIONS,
    FRAGILE_ROTATIONS,
    Rotation,
    apply_rotation,
    get_valid_rotations,
)


def test_all_rotations_count():
    assert len(ALL_ROTATIONS) == 6


def test_fragile_rotations_preserve_height():
    l, w, h = 3, 2, 4
    for rot in FRAGILE_ROTATIONS:
        _, _, rh = apply_rotation(l, w, h, rot)
        assert rh == h


def test_non_fragile_has_6_rotations():
    rots = get_valid_rotations(is_fragile=False)
    assert len(rots) == 6


def test_fragile_has_2_rotations():
    rots = get_valid_rotations(is_fragile=True)
    assert len(rots) == 2


def test_rotation_identity():
    l, w, h = 3, 2, 4
    rl, rw, rh = apply_rotation(
        l, w, h, Rotation.LWH
    )
    assert (rl, rw, rh) == (3, 2, 4)


def test_rotation_preserves_volume():
    l, w, h = 3, 2, 5
    vol = l * w * h
    for rot in ALL_ROTATIONS:
        rl, rw, rh = apply_rotation(l, w, h, rot)
        assert rl * rw * rh == vol


def test_all_rotations_produce_permutations():
    l, w, h = 3, 2, 5
    dims = sorted([l, w, h])
    for rot in ALL_ROTATIONS:
        result = sorted(
            apply_rotation(l, w, h, rot)
        )
        assert result == dims


def test_rotations_produce_distinct_orientations():
    l, w, h = 3, 2, 5
    orientations = set()
    for rot in ALL_ROTATIONS:
        orientations.add(
            apply_rotation(l, w, h, rot)
        )
    assert len(orientations) == 6
