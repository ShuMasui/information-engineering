"""Profit Sharing エージェント (表形式)。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lights_out.domain.agent.agent import Transition
from lights_out.domain.agent.tabular import TabularAgent, validate_epsilon_schedule


@dataclass(frozen=True)
class ProfitSharingConfig:
    """Profit Sharing のハイパーパラメータ。

    学習率と初期値は持たない (重みは 0 から始め、ゴール時の報酬をそのまま足す)。

    Attributes:
        discount_factor: 信用割当の減衰率 γ。ゴールの n 手前のルールには γⁿ 倍の報酬を配る。
            3×3・4×4 とも 0.01〜0.3 はほぼ同じ成績で、0.5 以上は遠回りが増え、0.9 以上は失敗も増えた。
            その範囲の中ほどの 0.1 を既定値にしている。
        epsilon_start: 探索率 ε の初期値。
        epsilon_min: ε の下限。
        epsilon_decay: エピソードごとに ε に掛ける係数 (1.0 なら減衰しない)。
        seed: 乱数シード。None なら非決定的。
    """

    discount_factor: float = 0.1
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.discount_factor <= 1.0:
            raise ValueError(f"discount_factor は (0, 1] にしてください: {self.discount_factor}")
        validate_epsilon_schedule(self.epsilon_start, self.epsilon_min, self.epsilon_decay)


class ProfitSharingAgent(TabularAgent):
    """ε-greedy 方策の Profit Sharing (ルール重みの強化学習)。

    ルール (盤面, 行動) ごとの重みを持ち、ゴールに着いたときだけ、そこまでの経路上のルールに
    報酬を配って強化する。1 手ごとの報酬や、ゴールできなかったエピソードは無視する。
    ゴールの n 手前のルールへは、ゴールの報酬の γⁿ 倍を配る (等比減少の信用割当関数)。
    ここでの γ は信用割当の減衰率で、合理性条件 (γ <= 1 / 行動数) はあくまで目安として扱う。
    """

    table_name = "weights"

    def __init__(self, n_actions: int, config: ProfitSharingConfig | None = None) -> None:
        self._config = config or ProfitSharingConfig()
        super().__init__(
            n_actions,
            epsilon_start=self._config.epsilon_start,
            epsilon_min=self._config.epsilon_min,
            epsilon_decay=self._config.epsilon_decay,
            initial_value=0.0,
            seed=self._config.seed,
        )
        self._episode: list[Transition] = []

    def update(self, transition: Transition) -> None:
        """重みは変えず、エピソードの終わりまで遷移を覚えておく (強化はゴールしてから)。"""
        self._episode.append(transition)

    def end_episode(self) -> None:
        """ゴールしていれば経路上の全ルールを強化し、覚えた遷移を捨てて ε を減衰させる。"""
        if self._episode and self._episode[-1].terminated:
            credit = self._episode[-1].reward
            for transition in reversed(self._episode):
                self._row(transition.state)[transition.action] += credit
                credit *= self._config.discount_factor
        self._episode.clear()
        super().end_episode()

    def hyperparameters(self) -> dict[str, Any]:
        return asdict(self._config)
