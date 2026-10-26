"""SARSA エージェント (表形式)。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lights_out.domain.agent.agent import Transition
from lights_out.domain.agent.tabular import TabularAgent, TabularConfig


@dataclass(frozen=True)
class SarsaConfig(TabularConfig):
    """SARSA のハイパーパラメータ (項目は ``TabularConfig`` と同じ)。"""


class SarsaAgent(TabularAgent):
    """ε-greedy 方策の表形式 SARSA (オン方策の TD 制御)。

    target に使う次の行動 a' は、次の ``update`` で初めて分かる。
    そのため遷移を 1 つ保留し、次の ``update`` か ``end_episode`` で更新する。
    """

    def __init__(self, n_actions: int, config: SarsaConfig | None = None) -> None:
        self._config = config or SarsaConfig()
        super().__init__(
            n_actions,
            epsilon_start=self._config.epsilon_start,
            epsilon_min=self._config.epsilon_min,
            epsilon_decay=self._config.epsilon_decay,
            initial_value=self._config.initial_q,
            seed=self._config.seed,
        )
        self._pending: Transition | None = None

    def update(self, transition: Transition) -> None:
        """保留中の遷移を、今回実際に取った行動を a' として更新し、今回の遷移を保留する。"""
        if self._pending is not None:
            # 今回の遷移はまだ更新していないので、Q(s,a') は教科書どおりの SARSA と同じ値になる
            self._learn(self._pending, transition.action)
        self._pending = transition

    def end_episode(self) -> None:
        """最後の遷移を更新してから ε を減衰させる (先に減衰すると、最後の a' だけ違う ε で選ぶことになる)。"""
        if self._pending is not None:
            self._learn(self._pending, None)
            self._pending = None
        super().end_episode()

    def hyperparameters(self) -> dict[str, Any]:
        return asdict(self._config)

    def _learn(self, transition: Transition, next_action: int | None) -> None:
        """Q(s,a) ← Q(s,a) + α (target − Q(s,a))。

        target は、クリアなら r、それ以外は r + γ Q(s',a')。
        打ち切りで終わった手は a' がまだないので、同じ ε-greedy 方策で選び、先の価値があるものとして更新する。
        """
        row = self._row(transition.state)
        target = transition.reward
        if not transition.terminated:
            if next_action is None:
                next_action = self.select_action(transition.next_state, explore=True)
            next_value = self._values(transition.next_state)[next_action]
            target += self._config.discount_factor * next_value
        row[transition.action] += self._config.learning_rate * (target - row[transition.action])
