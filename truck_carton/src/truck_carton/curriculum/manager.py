from __future__ import annotations

from collections import deque

import numpy as np

from truck_carton.config import (
    CurriculumConfig,
    CurriculumStage,
)


class CurriculumManager:
    """Manages progression through curriculum
    stages based on sustained reward thresholds."""

    def __init__(
        self, config: CurriculumConfig
    ) -> None:
        self._config = config
        self.current_stage: int = 0
        self._episode_rewards: deque[float] = deque(
            maxlen=config.stages[0].promotion_window
        )
        self.total_episodes: int = 0

    @property
    def stage(self) -> CurriculumStage:
        return self._config.stages[
            self.current_stage
        ]

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage >= (
            len(self._config.stages) - 1
        )

    @property
    def num_stages(self) -> int:
        return len(self._config.stages)

    @property
    def mean_reward(self) -> float:
        if not self._episode_rewards:
            return 0.0
        return float(np.mean(self._episode_rewards))

    def record_episode(self, reward: float) -> bool:
        self._episode_rewards.append(reward)
        self.total_episodes += 1

        if self.is_final_stage:
            return False

        if len(self._episode_rewards) >= (
            self.stage.promotion_window
        ):
            if (
                self.mean_reward
                >= self.stage.promotion_threshold
            ):
                return self._promote()

        return False

    def _promote(self) -> bool:
        if self.is_final_stage:
            return False

        self.current_stage += 1
        self._episode_rewards.clear()
        self._episode_rewards = deque(
            maxlen=self.stage.promotion_window
        )
        return True
