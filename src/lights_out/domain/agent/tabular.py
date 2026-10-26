"""表形式エージェントの共通部分 (ε-greedy の行動選択と、盤面ごとの値の表)。"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lights_out.domain.agent.agent import Agent


def validate_epsilon_schedule(start: float, minimum: float, decay: float) -> None:
    """探索率 ε の初期値・下限・減衰係数が正しい範囲か確かめる。"""
    if not 0.0 <= minimum <= start <= 1.0:
        raise ValueError(
            f"epsilon は 0 <= epsilon_min <= epsilon_start <= 1 にしてください: min={minimum}, start={start}"
        )
    if not 0.0 < decay <= 1.0:
        raise ValueError(f"epsilon_decay は (0, 1] にしてください: {decay}")


@dataclass(frozen=True)
class TabularConfig:
    """Q学習・SARSA・モンテカルロ法に共通のハイパーパラメータ。

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
        validate_epsilon_schedule(self.epsilon_start, self.epsilon_min, self.epsilon_decay)


class TabularAgent(Agent):
    """「盤面 (タプル) → 各行動の値」の辞書を持ち、ε-greedy で行動を選ぶエージェントの土台。

    表は必要になった盤面だけ作る (Q 学習系は訪れた盤面、Profit Sharing はゴールした経路上の盤面)。
    マス数が増えて状態数が爆発しても、実際に作った分しかメモリを使わない。
    サブクラスは ``update`` (と必要なら ``end_episode``) で表を更新する。
    ``end_episode`` は ε を減衰させるので、上書きするときは ``super().end_episode()`` も呼ぶ。
    """

    #: スナップショットのキー、および記録する表の大きさ (``<table_name>_size``) の名前。
    table_name = "q_table"

    def __init__(
        self,
        n_actions: int,
        *,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        initial_value: float,
        seed: int | None,
    ) -> None:
        super().__init__(n_actions)
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._initial_value = initial_value
        self._rng = random.Random(seed)
        self._epsilon = epsilon_start
        self._table: dict[tuple[int, ...], list[float]] = {}

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def select_action(self, state: Sequence[int], explore: bool = True) -> int:
        """explore=True は ε-greedy (最大の値が複数あればランダムに選ぶ)。

        explore=False は常に値が最大の行動 (同値なら番号の小さい方) で、乱数を使わない。
        """
        if explore and self._rng.random() < self._epsilon:
            return self._rng.randrange(self._n_actions)
        values = self._values(state)
        best = max(values)
        best_actions = [action for action, value in enumerate(values) if value == best]
        return self._rng.choice(best_actions) if explore else best_actions[0]

    def end_episode(self) -> None:
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def metrics(self) -> dict[str, float]:
        return {"epsilon": self._epsilon, f"{self.table_name}_size": float(len(self._table))}

    def snapshot(self) -> dict[str, Any]:
        return {
            "epsilon": self._epsilon,
            self.table_name: {",".join(map(str, state)): list(row) for state, row in self._table.items()},
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        table = {}
        for key, row in snapshot[self.table_name].items():
            if len(row) != self._n_actions:
                raise ValueError(f"表の値の数 {len(row)} が行動数 {self._n_actions} と一致しません")
            table[tuple(int(cell) for cell in key.split(","))] = list(row)
        self._epsilon = snapshot["epsilon"]
        self._table = table

    def _new_row(self) -> list[float]:
        return [self._initial_value] * self._n_actions

    def _row(self, state: Sequence[int]) -> list[float]:
        """state の各行動の値 (更新用)。未訪問なら初期値の行を表に加える。"""
        return self._table.setdefault(tuple(state), self._new_row())

    def _values(self, state: Sequence[int]) -> list[float]:
        """state の各行動の値 (参照用)。未訪問なら初期値で、表は変更しない。"""
        return self._table.get(tuple(state)) or self._new_row()
