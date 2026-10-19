"""実験 (学習 + 途中評価 + 記録) を実行するユースケース。"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from typing import Any

from lights_out.domain.agent import Agent
from lights_out.domain.environment import LightsOutEnvironment
from lights_out.domain.experiment import (
    EpisodeRecord,
    EvaluationRecord,
    ExperimentRepository,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentSettings:
    """実験の設定。

    Attributes:
        name: 実験の名前 (run のディレクトリ名に使うので、'/' や '\\' は含められない)。
        n_episodes: 学習するエピソード数。
        eval_interval: 何エピソードごとに評価するか。0 なら学習前と学習後だけ。
        eval_episodes: 1 回の評価で試すエピソード数。
        eval_seed: 評価用環境の乱数シード。毎回同じ開始盤面で比べられる。
    """

    name: str = "q-learning"
    n_episodes: int = 10000
    eval_interval: int = 1000
    eval_episodes: int = 100
    eval_seed: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip() or any(char in self.name for char in "/\\"):
            raise ValueError(f"name は空でなく、'/' や '\\' を含まない文字列にしてください: {self.name!r}")
        if self.n_episodes < 0:
            raise ValueError(f"n_episodes は 0 以上にしてください: {self.n_episodes}")
        if self.eval_interval < 0:
            raise ValueError(f"eval_interval は 0 以上にしてください: {self.eval_interval}")
        if self.eval_episodes < 1:
            raise ValueError(f"eval_episodes は 1 以上にしてください: {self.eval_episodes}")


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    episodes: list[EpisodeRecord]
    evaluations: list[EvaluationRecord]


class ExperimentRunner:
    """env と agent を受け取って学習し、設定と結果を repository に残す。"""

    def __init__(self, repository: ExperimentRepository) -> None:
        self._repository = repository

    def run(
        self, env: LightsOutEnvironment, agent: Agent, settings: ExperimentSettings
    ) -> ExperimentResult:
        run_id = self._repository.begin(settings.name, self._describe(env, agent, settings))
        logger.info("run %s を開始します", run_id)

        evaluations = [self._evaluate(env, agent, settings, episode=0)]

        def on_episode_end(record: EpisodeRecord) -> None:
            if settings.eval_interval and record.episode % settings.eval_interval == 0:
                evaluations.append(self._evaluate(env, agent, settings, episode=record.episode))

        episodes = agent.train(env, settings.n_episodes, on_episode_end)
        if evaluations[-1].episode != settings.n_episodes:
            evaluations.append(self._evaluate(env, agent, settings, episode=settings.n_episodes))

        self._repository.finish(
            run_id, episodes=episodes, evaluations=evaluations, agent_snapshot=agent.snapshot()
        )
        logger.info("run %s を保存しました", run_id)
        return ExperimentResult(run_id, episodes, evaluations)

    @staticmethod
    def _describe(
        env: LightsOutEnvironment, agent: Agent, settings: ExperimentSettings
    ) -> dict[str, Any]:
        """再現に必要な設定をまとめる。"""
        return {
            "settings": asdict(settings),
            "environment": asdict(env.config),
            "agent": {"type": type(agent).__name__, "hyperparameters": agent.hyperparameters()},
        }

    @staticmethod
    def _evaluate(
        env: LightsOutEnvironment, agent: Agent, settings: ExperimentSettings, *, episode: int
    ) -> EvaluationRecord:
        # 評価のたびに同じシードの環境を作り直すので、毎回同じ開始盤面で比べられる
        eval_env = LightsOutEnvironment(replace(env.config, seed=settings.eval_seed))
        summary = agent.evaluate(eval_env, settings.eval_episodes)
        logger.info(
            "episode %d: 成功率 %.1f%%, 平均手数 %.1f, 平均報酬 %.1f",
            episode,
            summary.success_rate * 100,
            summary.mean_steps,
            summary.mean_reward,
        )
        return EvaluationRecord(episode, summary)
