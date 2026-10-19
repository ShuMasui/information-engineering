"""エージェントの抽象クラス。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from lights_out.domain.environment import LightsOutEnvironment
from lights_out.domain.experiment import EpisodeRecord, EvaluationSummary


@dataclass(frozen=True)
class Transition:
    """環境との 1 回のやり取り (s, a, r, s')。

    ``terminated`` は本来の終了 (クリア) のみ。最大手数による打ち切りは含めない。
    """

    state: Sequence[int]
    action: int
    reward: float
    next_state: Sequence[int]
    terminated: bool


class Agent(ABC):
    """ライツアウトを解くエージェントの抽象クラス。

    サブクラスは行動選択 (``select_action``) と学習 (``update``)、および記録用の
    ``hyperparameters`` / ``snapshot`` / ``restore`` を実装する。
    エピソードを回す処理 (``play_episode`` / ``train`` / ``evaluate``) はここで共通化する。
    """

    def __init__(self, n_actions: int) -> None:
        if n_actions < 1:
            raise ValueError(f"n_actions は 1 以上にしてください: {n_actions}")
        self._n_actions = n_actions

    @property
    def n_actions(self) -> int:
        return self._n_actions

    # ----- サブクラスが実装する -----

    @abstractmethod
    def select_action(self, state: Sequence[int], explore: bool = True) -> int:
        """盤面 state に対する行動を選ぶ。explore=False なら探索せず、エージェントの状態も変えない。"""

    @abstractmethod
    def update(self, transition: Transition) -> None:
        """1 回のやり取りから学習する。"""

    @abstractmethod
    def hyperparameters(self) -> dict[str, Any]:
        """実験の記録に残すハイパーパラメータ (JSON にできる値)。"""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """学習結果を JSON にできる形で取り出す。"""

    @abstractmethod
    def restore(self, snapshot: dict[str, Any]) -> None:
        """``snapshot`` で取り出した学習結果を復元する。"""

    # ----- 必要なら上書きする -----

    def end_episode(self) -> None:
        """学習中のエピソード終了時に呼ばれる (探索率の減衰などに使う)。"""

    def metrics(self) -> dict[str, float]:
        """エピソードごとに記録したい値 (探索率など)。"""
        return {}

    # ----- 共通の処理 -----

    def play_episode(self, env: LightsOutEnvironment, *, learn: bool, episode: int = 1) -> EpisodeRecord:
        """env を reset して 1 エピソード進める。

        learn=True なら探索しながら各手で学習し、False なら探索も学習もせず方策どおりに動く。
        """
        state = env.reset()
        total_reward = 0.0
        while True:
            action = self.select_action(state, explore=learn)
            result = env.step(action)
            if learn:
                self.update(Transition(state, action, result.reward, result.state, result.terminated))
            total_reward += result.reward
            state = result.state
            if result.done:
                break
        if learn:
            self.end_episode()
        return EpisodeRecord(
            episode=episode,
            total_reward=total_reward,
            steps=env.steps,
            solved=result.terminated,
            metrics=self.metrics(),
        )

    def train(
        self,
        env: LightsOutEnvironment,
        n_episodes: int,
        on_episode_end: Callable[[EpisodeRecord], None] | None = None,
    ) -> list[EpisodeRecord]:
        """n_episodes エピソード学習し、エピソードごとの記録を返す。

        on_episode_end は 1 エピソードごとに記録を受け取る (途中評価やログ出力用)。
        """
        if n_episodes < 0:
            raise ValueError(f"n_episodes は 0 以上にしてください: {n_episodes}")
        records = []
        for episode in range(1, n_episodes + 1):
            record = self.play_episode(env, learn=True, episode=episode)
            records.append(record)
            if on_episode_end is not None:
                on_episode_end(record)
        return records

    def evaluate(self, env: LightsOutEnvironment, n_episodes: int) -> EvaluationSummary:
        """探索せず n_episodes エピソード試し、結果をまとめる。学習はしない。"""
        records = [
            self.play_episode(env, learn=False, episode=episode) for episode in range(1, n_episodes + 1)
        ]
        return EvaluationSummary.from_records(records)
