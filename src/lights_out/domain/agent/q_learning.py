"""Q学習エージェント (表形式)。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lights_out.domain.agent.agent import Transition
from lights_out.domain.agent.tabular import TabularAgent, TabularConfig


@dataclass(frozen=True)
class QLearningConfig(TabularConfig):
    """Q学習のハイパーパラメータ (項目は ``TabularConfig`` と同じ)。"""


class QLearningAgent(TabularAgent):
    """ε-greedy 方策の表形式 Q 学習 (オフ方策の TD 制御)。

    Q テーブルは「盤面 (タプル) → 各行動の Q 値」の辞書で、訪れた盤面だけ作る。
    """

    def __init__(self, n_actions: int, config: QLearningConfig | None = None) -> None:
        self._config = config or QLearningConfig()
        super().__init__(
            n_actions,
            epsilon_start=self._config.epsilon_start,
            epsilon_min=self._config.epsilon_min,
            epsilon_decay=self._config.epsilon_decay,
            initial_value=self._config.initial_q,
            seed=self._config.seed,
        )

    def update(self, transition: Transition) -> None:
        """Q(s,a) ← Q(s,a) + α (target − Q(s,a))。

        target は、クリアなら r、それ以外は r + γ max Q(s',·)。
        打ち切りで終わった手も、先の価値があるものとして後者で更新する。
        """
        row = self._row(transition.state)
        target = transition.reward
        if not transition.terminated:
            target += self._config.discount_factor * max(self._values(transition.next_state))
        row[transition.action] += self._config.learning_rate * (target - row[transition.action])

    def hyperparameters(self) -> dict[str, Any]:
        return asdict(self._config)
