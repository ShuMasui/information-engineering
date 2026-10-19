"""Q学習エージェント (表形式)。"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from lights_out.domain.agent.agent import Agent, Transition


@dataclass(frozen=True)
class QLearningConfig:
    """Q学習のハイパーパラメータ。

    Attributes:
        learning_rate: 学習率 α。
        discount_factor: 割引率 γ。
        epsilon_start: 探索率 ε の初期値。
        epsilon_min: ε の下限。
        epsilon_decay: エピソードごとに ε に掛ける係数 (1.0 なら減衰しない)。
        initial_q: Q 値の初期値。
        seed: 乱数シード。None なら非決定的。
    """

    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999
    initial_q: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError(f"learning_rate は (0, 1] にしてください: {self.learning_rate}")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError(f"discount_factor は [0, 1] にしてください: {self.discount_factor}")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon は 0 <= epsilon_min <= epsilon_start <= 1 にしてください: "
                f"min={self.epsilon_min}, start={self.epsilon_start}"
            )
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError(f"epsilon_decay は (0, 1] にしてください: {self.epsilon_decay}")


class QLearningAgent(Agent):
    """ε-greedy 方策の表形式 Q 学習。

    Q テーブルは「盤面 (タプル) → 各行動の Q 値」の辞書で、訪れた盤面だけ作る。
    マス数が増えて状態数が爆発しても、実際に訪れた分しかメモリを使わない。
    """

    def __init__(self, n_actions: int, config: QLearningConfig | None = None) -> None:
        super().__init__(n_actions)
        self._config = config or QLearningConfig()
        self._rng = random.Random(self._config.seed)
        self._epsilon = self._config.epsilon_start
        self._q_table: dict[tuple[int, ...], list[float]] = {}

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def select_action(self, state: Sequence[int], explore: bool = True) -> int:
        """explore=True は ε-greedy (最大の Q 値が複数あればランダムに選ぶ)。

        explore=False は常に Q 値が最大の行動 (同値なら番号の小さい方) で、乱数を使わない。
        """
        if explore and self._rng.random() < self._epsilon:
            return self._rng.randrange(self._n_actions)
        values = self._values(state)
        best = max(values)
        best_actions = [action for action, value in enumerate(values) if value == best]
        return self._rng.choice(best_actions) if explore else best_actions[0]

    def update(self, transition: Transition) -> None:
        """Q(s,a) ← Q(s,a) + α (target − Q(s,a))。

        target は、クリアなら r、それ以外は r + γ max Q(s',·)。
        打ち切りで終わった手も、先の価値があるものとして後者で更新する。
        """
        row = self._q_table.setdefault(tuple(transition.state), self._new_row())
        target = transition.reward
        if not transition.terminated:
            target += self._config.discount_factor * max(self._values(transition.next_state))
        row[transition.action] += self._config.learning_rate * (target - row[transition.action])

    def end_episode(self) -> None:
        self._epsilon = max(self._config.epsilon_min, self._epsilon * self._config.epsilon_decay)

    def metrics(self) -> dict[str, float]:
        return {"epsilon": self._epsilon, "q_table_size": float(len(self._q_table))}

    def hyperparameters(self) -> dict[str, Any]:
        return asdict(self._config)

    def snapshot(self) -> dict[str, Any]:
        return {
            "epsilon": self._epsilon,
            "q_table": {",".join(map(str, state)): list(row) for state, row in self._q_table.items()},
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        q_table = {}
        for key, row in snapshot["q_table"].items():
            if len(row) != self._n_actions:
                raise ValueError(f"Q 値の数 {len(row)} が行動数 {self._n_actions} と一致しません")
            q_table[tuple(int(cell) for cell in key.split(","))] = list(row)
        self._epsilon = snapshot["epsilon"]
        self._q_table = q_table

    def _new_row(self) -> list[float]:
        return [self._config.initial_q] * self._n_actions

    def _values(self, state: Sequence[int]) -> list[float]:
        """state の各行動の Q 値。未訪問なら初期値で、テーブルは変更しない。"""
        return self._q_table.get(tuple(state)) or self._new_row()
