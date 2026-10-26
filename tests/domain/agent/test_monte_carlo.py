import json

import pytest

from lights_out.domain.agent import MonteCarloAgent, MonteCarloConfig, Transition
from lights_out.domain.environment import EnvironmentConfig, LightsOutEnvironment

S0 = (0, 1)
S1 = (1, 0)
S2 = (1, 1)


def make_agent(**kwargs) -> MonteCarloAgent:
    kwargs.setdefault("seed", 0)
    return MonteCarloAgent(n_actions=3, config=MonteCarloConfig(**kwargs))


def q_row(agent: MonteCarloAgent, state) -> list[float]:
    return agent.snapshot()["q_table"][",".join(map(str, state))]


def set_q(agent: MonteCarloAgent, state, values: list[float]) -> None:
    snapshot = agent.snapshot()
    snapshot["q_table"][",".join(map(str, state))] = values
    agent.restore(snapshot)


# ----- エピソード中は記録するだけ -----


def test_update_alone_does_not_touch_the_table():
    agent = make_agent(learning_rate=1.0)
    set_q(agent, S0, [1.0, 2.0, 3.0])
    before = agent.snapshot()
    agent.update(Transition(S0, 1, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 0, 100.0, S2, terminated=True))
    assert agent.snapshot() == before
    assert agent.metrics()["q_table_size"] == 1.0  # S1 は表に加わっていない


# ----- エピソード終了時の更新 -----


def test_end_episode_moves_q_toward_the_return():
    agent = make_agent(learning_rate=0.5)
    set_q(agent, S0, [0.0, 4.0, 0.0])
    agent.update(Transition(S0, 1, 10.0, S1, terminated=True))
    agent.end_episode()
    # G = 10, Q = 4 + 0.5 * (10 - 4)
    assert q_row(agent, S0) == [0.0, pytest.approx(7.0), 0.0]


def test_returns_are_discounted_back_from_the_end_of_the_episode():
    agent = make_agent(learning_rate=0.5, discount_factor=0.9)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 1, -1.0, S2, terminated=False))
    agent.update(Transition(S2, 2, 100.0, (0, 0), terminated=True))
    agent.end_episode()
    # G2 = 100, G1 = -1 + 0.9 * 100 = 89, G0 = -1 + 0.9 * 89 = 79.1。Q = 0.5 * G
    assert q_row(agent, S2) == [0.0, 0.0, pytest.approx(50.0)]
    assert q_row(agent, S1) == [0.0, pytest.approx(44.5), 0.0]
    assert q_row(agent, S0) == [pytest.approx(39.55), 0.0, 0.0]


def test_a_repeated_pair_is_updated_once_with_the_return_of_its_first_visit():
    agent = make_agent(learning_rate=0.5, discount_factor=0.5)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 1, -2.0, S0, terminated=False))
    agent.update(Transition(S0, 0, -3.0, S0, terminated=False))  # (S0, 0) の 2 回目
    agent.update(Transition(S0, 1, 8.0, S2, terminated=True))  # 同じ盤面でも別の行動
    agent.end_episode()
    # G3 = 8, G2 = -3 + 0.5 * 8 = 1, G1 = -2 + 0.5 * 1 = -1.5, G0 = -1 + 0.5 * (-1.5) = -1.75
    # (S0, 0) は 1 回目の G0 だけで 0.5 * (-1.75)。2 回目の G2 では更新しない
    assert q_row(agent, S0) == [pytest.approx(-0.875), pytest.approx(4.0), 0.0]
    assert q_row(agent, S1) == [0.0, pytest.approx(-0.75), 0.0]


@pytest.mark.parametrize("terminated", [True, False])
def test_episodes_do_not_leak_into_each_other(terminated):
    # クリアで終わっても打ち切りで終わっても、次のエピソードに前の記録を持ち越さない
    agent = make_agent(learning_rate=0.5, discount_factor=0.9)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=terminated))
    agent.end_episode()
    assert q_row(agent, S0) == [-0.5, 0.0, 0.0]

    agent.update(Transition(S1, 1, 10.0, S2, terminated=terminated))
    agent.end_episode()
    # 2 回目は S1 の分だけ。S0 を更新し直さない
    assert q_row(agent, S1) == [0.0, 5.0, 0.0]
    assert q_row(agent, S0) == [-0.5, 0.0, 0.0]


def test_list_and_tuple_states_are_the_same_board():
    agent = make_agent(learning_rate=1.0, discount_factor=0.5)
    agent.update(Transition([0, 1], 0, -1.0, [1, 0], terminated=False))
    agent.update(Transition([1, 0], 1, 10.0, [0, 1], terminated=False))
    agent.update(Transition((0, 1), 0, -5.0, (1, 0), terminated=True))  # (S0, 0) の 2 回目
    agent.end_episode()
    # G2 = -5, G1 = 10 + 0.5 * (-5) = 7.5, G0 = -1 + 0.5 * 7.5 = 2.75
    assert q_row(agent, S0) == [pytest.approx(2.75), 0.0, 0.0]
    assert q_row(agent, S1) == [0.0, pytest.approx(7.5), 0.0]


def test_initial_q_is_the_starting_point_of_the_update():
    agent = make_agent(learning_rate=0.5, initial_q=5.0)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=True))
    agent.end_episode()
    assert q_row(agent, S0) == [2.0, 5.0, 5.0]  # 5 + 0.5 * (-1 - 5)


def test_a_truncated_episode_does_not_bootstrap_from_the_next_state():
    # 打ち切りの最後の手 (terminated=False) でも、その先の Q 値は使わず G = 0 として扱う
    agent = make_agent(learning_rate=1.0, discount_factor=0.5)
    set_q(agent, S2, [100.0, 100.0, 100.0])
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 1, -1.0, S2, terminated=False))
    agent.end_episode()
    # G1 = -1, G0 = -1 + 0.5 * (-1) = -1.5。S2 の大きな Q 値は混ざらない
    assert q_row(agent, S1) == [0.0, pytest.approx(-1.0), 0.0]
    assert q_row(agent, S0) == [pytest.approx(-1.5), 0.0, 0.0]


def test_end_episode_decays_epsilon_down_to_the_minimum():
    agent = make_agent(epsilon_start=1.0, epsilon_min=0.3, epsilon_decay=0.5)
    seen = []
    for _ in range(4):
        agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
        agent.end_episode()
        seen.append(agent.epsilon)
    assert seen == [0.5, 0.3, 0.3, 0.3]


def test_end_episode_without_updates_only_decays_epsilon():
    agent = make_agent(epsilon_start=1.0, epsilon_decay=0.5)
    agent.end_episode()
    assert agent.epsilon == 0.5
    assert agent.snapshot()["q_table"] == {}


# ----- 記録 -----


def test_hyperparameters_are_json_serializable_and_complete():
    config = MonteCarloConfig(learning_rate=0.2, seed=3)
    params = MonteCarloAgent(4, config).hyperparameters()
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


def test_metrics_report_epsilon_and_table_size_once_the_episode_ends():
    agent = make_agent(epsilon_start=0.8, epsilon_decay=0.5)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=True))
    assert agent.metrics() == {"epsilon": 0.8, "q_table_size": 0.0}
    agent.end_episode()
    assert agent.metrics() == {"epsilon": 0.4, "q_table_size": 1.0}


# ----- 行動選択 -----


def test_selection_follows_the_table_only_after_the_episode_ends():
    agent = make_agent(learning_rate=1.0, epsilon_start=0.0, epsilon_min=0.0)
    agent.update(Transition(S0, 2, 5.0, S1, terminated=True))
    assert agent.select_action(S0, explore=False) == 0  # エピソード中はまだ表が空 (同値なら番号の小さい方)
    agent.end_episode()
    assert all(agent.select_action(S0) == 2 for _ in range(20))
    assert agent.select_action(S0, explore=False) == 2


def test_selection_without_exploration_changes_nothing():
    agent = make_agent()
    set_q(agent, S0, [0.0, 3.0, 3.0])
    before = agent.snapshot()
    state_before = agent._rng.getstate()
    assert {agent.select_action(S0, explore=False) for _ in range(10)} == {1}
    assert agent.snapshot() == before
    assert agent._rng.getstate() == state_before


# ----- 保存・復元 -----


def test_snapshot_is_json_serializable_and_restores_to_an_equal_agent():
    agent = make_agent(learning_rate=0.5)
    agent.update(Transition(S0, 1, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 2, 10.0, S0, terminated=True))
    agent.end_episode()
    assert q_row(agent, S0)[1] != 0.0

    snapshot = json.loads(json.dumps(agent.snapshot()))
    restored = make_agent()
    restored.restore(snapshot)

    assert restored.snapshot() == agent.snapshot()
    assert restored.epsilon == agent.epsilon
    for state in (S0, S1):
        assert restored.select_action(state, explore=False) == agent.select_action(state, explore=False)


# ----- 再現性 -----


def test_same_seed_gives_same_actions():
    a, b = make_agent(seed=5), make_agent(seed=5)
    assert [a.select_action(S0) for _ in range(30)] == [b.select_action(S0) for _ in range(30)]


def test_same_seed_gives_same_training_result():
    def train(seed: int) -> dict:
        env = LightsOutEnvironment(EnvironmentConfig(size=3, scramble_moves=2, max_steps=8, seed=seed))
        agent = MonteCarloAgent(env.n_actions, MonteCarloConfig(seed=seed))
        agent.train(env, n_episodes=30)
        return agent.snapshot()

    assert train(4) == train(4)
    assert train(4) != train(5)


# ----- 設定の検証 -----


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"discount_factor": 1.1},
        {"epsilon_start": 0.1, "epsilon_min": 0.5},
        {"epsilon_decay": 0.0},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        MonteCarloConfig(**kwargs)


# ----- 実際に学習できる -----


def test_a_training_episode_updates_the_table_when_it_is_over():
    env = LightsOutEnvironment(EnvironmentConfig(size=3, scramble_moves=1, max_steps=5, seed=0))
    agent = MonteCarloAgent(env.n_actions, MonteCarloConfig(epsilon_decay=0.5, seed=0))
    record = agent.play_episode(env, learn=True)
    assert record.metrics["q_table_size"] >= 1.0
    assert record.metrics["epsilon"] == 0.5


def test_agent_learns_to_clear_boards_one_press_away():
    def make_env(seed: int) -> LightsOutEnvironment:
        return LightsOutEnvironment(
            EnvironmentConfig(size=3, scramble_moves=1, max_steps=10, seed=seed)
        )

    train_env, eval_env = make_env(0), make_env(9)
    # 初期値 0 だと、ε が下がった後は「間違えた手から最短で戻る遠回り (収益 ≈ 96)」に居座り、
    # 1 手で解ける手 (収益 100) をほとんど試さないことが多い。楽観的な初期値 (= クリア報酬) で全部の手を試させる
    agent = MonteCarloAgent(
        train_env.n_actions,
        MonteCarloConfig(initial_q=100.0, epsilon_decay=0.99, epsilon_min=0.05, seed=1),
    )
    assert agent.evaluate(make_env(9), n_episodes=50).success_rate < 0.5

    agent.train(train_env, n_episodes=1500)

    summary = agent.evaluate(eval_env, n_episodes=50)
    assert summary.success_rate == 1.0
    assert summary.mean_steps == 1.0
