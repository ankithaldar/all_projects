from truck_carton.config import (
    CurriculumConfig,
    CurriculumStage,
)
from truck_carton.curriculum.manager import (
    CurriculumManager,
)


def _make_config(
    threshold: float = 0.5, window: int = 3
) -> CurriculumConfig:
    return CurriculumConfig(stages=(
        CurriculumStage(
            'toy', 2, 2, 10, threshold, window
        ),
        CurriculumStage(
            'small', 3, 3, 20, threshold, window
        ),
        CurriculumStage(
            'medium', 5, 4, 40, threshold, window
        ),
    ))


def test_initial_state():
    cm = CurriculumManager(_make_config())
    assert cm.current_stage == 0
    assert cm.stage.name == 'toy'
    assert cm.is_final_stage is False
    assert cm.total_episodes == 0


def test_no_promotion_below_threshold():
    cm = CurriculumManager(
        _make_config(threshold=0.8, window=3)
    )
    for _ in range(10):
        promoted = cm.record_episode(0.5)
        assert promoted is False
    assert cm.current_stage == 0


def test_promotion_above_threshold():
    cm = CurriculumManager(
        _make_config(threshold=0.5, window=3)
    )
    cm.record_episode(0.6)
    cm.record_episode(0.7)
    promoted = cm.record_episode(0.8)
    assert promoted is True
    assert cm.current_stage == 1
    assert cm.stage.name == 'small'


def test_double_promotion():
    cm = CurriculumManager(
        _make_config(threshold=0.5, window=2)
    )
    cm.record_episode(0.9)
    cm.record_episode(0.9)
    assert cm.current_stage == 1

    cm.record_episode(0.9)
    promoted = cm.record_episode(0.9)
    assert promoted is True
    assert cm.current_stage == 2
    assert cm.stage.name == 'medium'


def test_final_stage_no_promotion():
    cm = CurriculumManager(
        _make_config(threshold=0.5, window=2)
    )
    cm.record_episode(0.9)
    cm.record_episode(0.9)
    cm.record_episode(0.9)
    cm.record_episode(0.9)
    assert cm.current_stage == 2
    assert cm.is_final_stage is True

    promoted = cm.record_episode(0.9)
    assert promoted is False
    assert cm.current_stage == 2


def test_mean_reward():
    cm = CurriculumManager(
        _make_config(window=3)
    )
    cm.record_episode(0.3)
    cm.record_episode(0.6)
    cm.record_episode(0.9)
    assert abs(cm.mean_reward - 0.6) < 1e-9


def test_total_episodes():
    cm = CurriculumManager(_make_config())
    for _ in range(5):
        cm.record_episode(0.1)
    assert cm.total_episodes == 5
