"""実験の記録 (値オブジェクト) と、その保存先のインターフェース。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EpisodeRecord:
    """1 エピソードの結果。

    Attributes:
        episode: エピソード番号 (1 始まり)。
        total_reward: そのエピソードの報酬の合計。
        steps: 押した回数。
        solved: クリアできたか。
        metrics: エージェントが記録したい値 (探索率など)。エピソード終了時点のもの。
    """

    episode: int
    total_reward: float
    steps: int
    solved: bool
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSummary:
    """複数エピソードの評価結果。"""

    n_episodes: int
    success_rate: float
    mean_reward: float
    mean_steps: float

    @classmethod
    def from_records(cls, records: Sequence[EpisodeRecord]) -> EvaluationSummary:
        if not records:
            raise ValueError("評価するエピソードがありません")
        n = len(records)
        return cls(
            n_episodes=n,
            success_rate=sum(r.solved for r in records) / n,
            mean_reward=sum(r.total_reward for r in records) / n,
            mean_steps=sum(r.steps for r in records) / n,
        )


@dataclass(frozen=True)
class EvaluationRecord:
    """学習中のある時点 (episode 回学習した後) での評価結果。0 なら学習前。"""

    episode: int
    summary: EvaluationSummary


class ExperimentRepository(ABC):
    """実験 (1 回の実行 = run) の記録を保存する場所。"""

    @abstractmethod
    def begin(self, name: str, config: dict[str, Any]) -> str:
        """run を開始して設定を保存し、run の ID を返す。"""

    @abstractmethod
    def finish(
        self,
        run_id: str,
        *,
        episodes: Sequence[EpisodeRecord],
        evaluations: Sequence[EvaluationRecord],
        agent_snapshot: dict[str, Any],
    ) -> None:
        """run の結果 (エピソードごとの記録・評価・学習済みエージェント) を保存する。"""
