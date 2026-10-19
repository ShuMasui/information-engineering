import json

import pytest

from lights_out.domain.agent import QLearningAgent, QLearningConfig, Transition
from lights_out.domain.environment import EnvironmentConfig, LightsOutEnvironment

S0 = (0, 1)
S1 = (1, 0)


def make_agent(**kwargs) -> QLearningAgent:
    kwargs.setdefault("seed", 0)
    return QLearningAgent(n_actions=3, config=QLearningConfig(**kwargs))


def q_row(agent: QLearningAgent, state) -> list[float]:
    return agent.snapshot()["q_table"][",".join(map(str, state))]


def set_q(agent: QLearningAgent, state, values: list[float]) -> None:
    snapshot = agent.snapshot()
    snapshot["q_table"][",".join(map(str, state))] = values
    agent.restore(snapshot)


# ----- 更新式 -----


def test_update_moves_q_toward_the_td_target():
    agent = make_agent(learning_rate=0.5, discount_factor=0.9)
    set_q(agent, S1, [2.0, 4.0, 1.0])
    agent.update(Transition(S0, 1, -1.0, S1, terminated=False))
    # target = -1 + 0.9 * 4 = 2.6, Q = 0 + 0.5 * 2.6
    assert q_row(agent, S0) == [0.0, pytest.approx(1.3), 0.0]


def test_update_ignores_next_state_when_terminated():
    agent = make_agent(learning_rate=0.5, discount_factor=0.9)
    set_q(agent, S1, [100.0, 100.0, 100.0])
    agent.update(Transition(S0, 2, 10.0, S1, terminated=True))
    assert q_row(agent, S0) == [0.0, 0.0, pytest.approx(5.0)]


def test_update_bootstraps_when_the_episode_was_only_truncated():
    # 打ち切りは Transition.terminated=False で渡る。先の価値を使って更新する
    agent = make_agent(learning_rate=1.0, discount_factor=0.5)
    set_q(agent, S1, [8.0, 0.0, 0.0])
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    assert q_row(agent, S0)[0] == pytest.approx(3.0)


def test_update_only_changes_the_chosen_action():
    agent = make_agent(learning_rate=1.0)
    agent.update(Transition(S0, 1, -1.0, S1, terminated=True))
    assert q_row(agent, S0) == [0.0, -1.0, 0.0]


def test_update_accepts_list_states():
    agent = make_agent(learning_rate=1.0)
    agent.update(Transition([0, 1], 0, -1.0, [1, 0], terminated=True))
    assert q_row(agent, S0)[0] == -1.0


def test_initial_q_is_used_for_unvisited_states():
    agent = make_agent(learning_rate=1.0, discount_factor=1.0, initial_q=5.0)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    assert q_row(agent, S0) == [4.0, 5.0, 5.0]  # -1 + 1 * max(初期値 5)


# ----- 行動選択 -----


def test_greedy_selection_picks_the_largest_q():
    agent = make_agent(epsilon_start=0.0, epsilon_min=0.0)
    set_q(agent, S0, [0.0, 3.0, 1.0])
    assert all(agent.select_action(S0) == 1 for _ in range(20))


def test_epsilon_one_selects_every_action_eventually():
    agent = make_agent(epsilon_start=1.0)
    set_q(agent, S0, [0.0, 3.0, 1.0])
    assert {agent.select_action(S0) for _ in range(200)} == {0, 1, 2}


def test_exploring_greedy_breaks_ties_randomly():
    agent = make_agent(epsilon_start=0.0, epsilon_min=0.0)
    assert {agent.select_action(S0) for _ in range(200)} == {0, 1, 2}


def test_exploring_greedy_ties_only_among_the_best():
    agent = make_agent(epsilon_start=0.0, epsilon_min=0.0)
    set_q(agent, S0, [1.0, 0.0, 1.0])
    assert {agent.select_action(S0) for _ in range(200)} == {0, 2}


def test_without_exploration_selection_is_greedy_and_deterministic():
    agent = make_agent(epsilon_start=1.0)
    set_q(agent, S0, [0.0, 3.0, 3.0])
    assert {agent.select_action(S0, explore=False) for _ in range(50)} == {1}
    assert {agent.select_action((9, 9), explore=False) for _ in range(50)} == {0}  # 未訪問は同値 → 番号の小さい方


def test_selection_without_exploration_changes_nothing():
    agent = make_agent()
    before = agent.snapshot()
    state_before = agent._rng.getstate()
    for _ in range(10):
        agent.select_action(S0, explore=False)
    assert agent.snapshot() == before
    assert agent._rng.getstate() == state_before


def test_selection_does_not_add_unvisited_states_to_the_table():
    agent = make_agent()
    agent.select_action(S0)
    assert agent.snapshot()["q_table"] == {}


# ----- ε の減衰・記録 -----


def test_end_episode_decays_epsilon_down_to_the_minimum():
    agent = make_agent(epsilon_start=1.0, epsilon_min=0.3, epsilon_decay=0.5)
    seen = []
    for _ in range(4):
        agent.end_episode()
        seen.append(agent.epsilon)
    assert seen == [0.5, 0.3, 0.3, 0.3]


def test_metrics_report_epsilon_and_table_size():
    agent = make_agent(epsilon_start=0.8)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=True))
    assert agent.metrics() == {"epsilon": 0.8, "q_table_size": 1.0}


def test_hyperparameters_are_json_serializable_and_complete():
    config = QLearningConfig(learning_rate=0.2, seed=3)
    params = QLearningAgent(4, config).hyperparameters()
    assert json.loads(json.dumps(params)) == params
    assert params["learning_rate"] == 0.2
    assert params["seed"] == 3
    assert set(params) == {
        "learning_rate",
        "discount_factor",
        "epsilon_start",
        "epsilon_min",
        "epsilon_decay",
        "initial_q",
        "seed",
    }


# ----- 保存・復元 -----


def test_snapshot_is_json_serializable_and_restores_to_an_equal_agent():
    agent = make_agent(learning_rate=0.5)
    agent.update(Transition(S0, 1, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 2, 10.0, S0, terminated=True))
    agent.end_episode()

    snapshot = json.loads(json.dumps(agent.snapshot()))
    restored = make_agent()
    restored.restore(snapshot)

    assert restored.snapshot() == agent.snapshot()
    assert restored.epsilon == agent.epsilon
    for state in (S0, S1):
        assert restored.select_action(state, explore=False) == agent.select_action(state, explore=False)


def test_restore_rejects_rows_of_the_wrong_length():
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.restore({"epsilon": 0.1, "q_table": {"0,1": [0.0, 0.0]}})


# ----- 再現性 -----


def test_same_seed_gives_same_actions():
    a, b = make_agent(seed=5), make_agent(seed=5)
    assert [a.select_action(S0) for _ in range(30)] == [b.select_action(S0) for _ in range(30)]


# ----- 設定の検証 -----


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"learning_rate": 1.5},
        {"discount_factor": -0.1},
        {"discount_factor": 1.1},
        {"epsilon_start": 1.5},
        {"epsilon_start": 0.1, "epsilon_min": 0.5},
        {"epsilon_min": -0.1},
        {"epsilon_decay": 0.0},
        {"epsilon_decay": 1.1},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        QLearningConfig(**kwargs)


# ----- 実際に学習できる -----


def test_agent_learns_to_clear_boards_one_press_away():
    def make_env(seed: int) -> LightsOutEnvironment:
        return LightsOutEnvironment(
            EnvironmentConfig(size=3, scramble_moves=1, max_steps=10, seed=seed)
        )

    train_env, eval_env = make_env(0), make_env(9)
    agent = QLearningAgent(
        train_env.n_actions, QLearningConfig(epsilon_decay=0.99, epsilon_min=0.05, seed=1)
    )
    assert agent.evaluate(make_env(9), n_episodes=50).success_rate < 0.5

    agent.train(train_env, n_episodes=1500)

    summary = agent.evaluate(eval_env, n_episodes=50)
    assert summary.success_rate == 1.0
    assert summary.mean_steps == 1.0
