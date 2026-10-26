"""モンテカルロ法エージェント (表形式)。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lights_out.domain.agent.agent import Transition
from lights_out.domain.agent.tabular import TabularAgent, TabularConfig


@dataclass(frozen=True)
class MonteCarloConfig(TabularConfig):
    """モンテカルロ法のハイパーパラメータ (項目は ``TabularConfig`` と同じ)。"""


class MonteCarloAgent(TabularAgent):
    """ε-greedy 方策の表形式モンテカルロ制御 (初回訪問・オン方策・一定の学習率)。

    エピソードが終わるまで表は更新せず、終わってから実際の割引収益 G で更新する。
    ブートストラップしないので、最大手数による打ち切りもクリアと同じくエピソードの終わり
    (その先の収益は 0) として扱う。Q学習・SARSA とはここが意図的に違う。
    """

    def __init__(self, n_actions: int, config: MonteCarloConfig | None = None) -> None:
        self._config = config or MonteCarloConfig()
        super().__init__(
            n_actions,
            epsilon_start=self._config.epsilon_start,
            epsilon_min=self._config.epsilon_min,
            epsilon_decay=self._config.epsilon_decay,
            initial_value=self._config.initial_q,
            seed=self._config.seed,
        )
        self._episode: list[Transition] = []

    def update(self, transition: Transition) -> None:
        """収益は最後まで分からないので、エピソード中は記録するだけで表は変えない。"""
        self._episode.append(transition)

    def end_episode(self) -> None:
        """Q(s,a) ← Q(s,a) + α (G − Q(s,a))。各 (盤面, 行動) の最初の出現だけを更新する。

        G は、その手から最後までの割引収益 (G_t = r_t + γ G_{t+1}、最後の手の先は 0)。
        更新のあと記録を空にして、最後に ε を減衰させる。
        """
        first_returns: dict[tuple[tuple[int, ...], int], float] = {}
        g = 0.0
        for transition in reversed(self._episode):
            g = transition.reward + self._config.discount_factor * g
            # 後ろから辿るので、同じ組が再び出てきたら上書きされ、最初の出現の収益が残る
            first_returns[(tuple(transition.state), transition.action)] = g
        for (state, action), g in first_returns.items():
            row = self._row(state)
            row[action] += self._config.learning_rate * (g - row[action])
        self._episode.clear()
        super().end_episode()

    def hyperparameters(self) -> dict[str, Any]:
        return asdict(self._config)
