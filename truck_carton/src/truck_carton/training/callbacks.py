from __future__ import annotations

from stable_baselines3.common.callbacks import (
    BaseCallback,
)

from truck_carton.curriculum.manager import (
    CurriculumManager,
)


class CurriculumCallback(BaseCallback):
    """Monitors episode rewards and promotes
    curriculum stages when thresholds are met."""

    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._curriculum = curriculum_manager

    def _on_step(self) -> bool:
        infos = self.locals.get('infos')
        if not infos:
            return True

        for info in infos:
            ep = info.get('episode')
            if ep is None:
                continue

            episode_reward = ep['r']
            promoted = self._curriculum.record_episode(
                episode_reward
            )

            if promoted:
                env = self.training_env.envs[0]
                while hasattr(env, 'env'):
                    env = env.env
                env.set_curriculum_stage(
                    self._curriculum.current_stage
                )

                self.logger.record(
                    'curriculum/stage',
                    self._curriculum.current_stage,
                )
                self.logger.record(
                    'curriculum/stage_name',
                    self._curriculum.stage.name,
                )
                if self.verbose > 0:
                    print(
                        '[Curriculum] Promoted to'
                        f' stage'
                        f' {self._curriculum.current_stage}:'
                        f' {self._curriculum.stage.name}'
                    )

        return True


class MetricsCallback(BaseCallback):
    """Logs per-component reward metrics to
    TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos')
        if not infos:
            return True

        for info in infos:
            breakdown = info.get(
                'reward_breakdown'
            )
            if breakdown:
                for name, value in (
                    breakdown.items()
                ):
                    self.logger.record(
                        f'reward/{name}', value
                    )

            num_placed = info.get('num_placed')
            if num_placed is not None:
                self.logger.record(
                    'env/num_placed', num_placed
                )

        return True
