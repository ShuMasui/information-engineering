"""実験で保存した agent.json から、手法ごとに学習結果を復元できることを実ファイルで確かめる。"""

import json

import pytest

from lights_out.application import ExperimentRunner, ExperimentSettings
from lights_out.domain.agent import (
    MonteCarloAgent,
    MonteCarloConfig,
    ProfitSharingAgent,
    ProfitSharingConfig,
    QLearningAgent,
    QLearningConfig,
    SarsaAgent,
    SarsaConfig,
)
from lights_out.domain.environment import EnvironmentConfig, LightsOutEnvironment
from lights_out.infrastructure import FileExperimentRepository

AGENTS = [
    pytest.param(QLearningAgent, QLearningConfig, id="q-learning"),
    pytest.param(SarsaAgent, SarsaConfig, id="sarsa"),
    pytest.param(ProfitSharingAgent, ProfitSharingConfig, id="profit-sharing"),
    pytest.param(MonteCarloAgent, MonteCarloConfig, id="monte-carlo"),
]


@pytest.mark.parametrize(("agent_cls", "config_cls"), AGENTS)
def test_saved_agent_restores_into_a_fresh_agent_with_the_same_policy(tmp_path, agent_cls, config_cls):
    env = LightsOutEnvironment(EnvironmentConfig(size=3, scramble_moves=1, max_steps=10, seed=0))
    trained = agent_cls(env.n_actions, config_cls(seed=1))
    settings = ExperimentSettings(n_episodes=300, eval_interval=0, eval_episodes=5)
    result = ExperimentRunner(FileExperimentRepository(tmp_path)).run(env, trained, settings)

    saved = json.loads((tmp_path / result.run_id / "agent.json").read_text(encoding="utf-8"))
    fresh = agent_cls(env.n_actions, config_cls(seed=99))  # 乱数は復元の対象ではない
    fresh.restore(saved)

    assert fresh.snapshot() == trained.snapshot()
    visited = [tuple(int(cell) for cell in key.split(",")) for key in saved[agent_cls.table_name]]
    assert len(visited) > 1
    for state in visited:
        assert fresh.select_action(state, explore=False) == trained.select_action(state, explore=False)
    # 学習した結果が入っている (全部初期値のままの表を復元しているだけではない)
    assert any(any(value != 0.0 for value in row) for row in saved[agent_cls.table_name].values())
