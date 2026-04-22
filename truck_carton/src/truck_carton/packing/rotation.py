from enum import IntEnum


class Rotation(IntEnum):
    LWH = 0
    LHW = 1
    WLH = 2
    WHL = 3
    HLW = 4
    HWL = 5


_ROTATION_PERMUTATIONS: dict[Rotation, tuple[int, int, int]] = {
    Rotation.LWH: (0, 1, 2),
    Rotation.LHW: (0, 2, 1),
    Rotation.WLH: (1, 0, 2),
    Rotation.WHL: (1, 2, 0),
    Rotation.HLW: (2, 0, 1),
    Rotation.HWL: (2, 1, 0),
}

FRAGILE_ROTATIONS: tuple[Rotation, ...] = (
    Rotation.LWH,
    Rotation.WLH,
)
ALL_ROTATIONS: tuple[Rotation, ...] = tuple(Rotation)


def apply_rotation(
    l: int, w: int, h: int, rotation: Rotation
) -> tuple[int, int, int]:
    dims = (l, w, h)
    perm = _ROTATION_PERMUTATIONS[rotation]
    return (dims[perm[0]], dims[perm[1]], dims[perm[2]])


def get_valid_rotations(
    is_fragile: bool,
) -> tuple[Rotation, ...]:
    return FRAGILE_ROTATIONS if is_fragile else ALL_ROTATIONS
