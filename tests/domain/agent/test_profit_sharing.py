import json

import pytest

from lights_out.domain.agent import ProfitSharingAgent, ProfitSharingConfig, Transition
from lights_out.domain.environment import EnvironmentConfig, LightsOutEnvironment

S0 = (0, 1)
S1 = (1, 0)
S2 = (1, 1)


def make_agent(**kwargs) -> ProfitSharingAgent:
    kwargs.setdefault("seed", 0)
    return ProfitSharingAgent(n_actions=3, config=ProfitSharingConfig(**kwargs))


def weight_row(agent: ProfitSharingAgent, state) -> list[float]:
    return agent.snapshot()["weights"][",".join(map(str, state))]


def set_weights(agent: ProfitSharingAgent, state, values: list[float]) -> None:
    snapshot = agent.snapshot()
    snapshot["weights"][",".join(map(str, state))] = values
    agent.restore(snapshot)


# ----- 強化は end_episode まで行わない -----


def test_update_alone_does_not_change_weights():
    agent = make_agent()
    agent.update(Transition(S0, 1, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 2, 100.0, S2, terminated=True))
    assert agent.snapshot()["weights"] == {}


# ----- ゴール時の強化 -----


def test_reaching_the_goal_reinforces_each_rule_by_goal_reward_times_gamma_to_the_n():
    agent = make_agent(discount_factor=0.5)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))  # ゴールの 2 手前
    agent.update(Transition(S1, 2, -1.0, S2, terminated=False))  # ゴールの 1 手前
    agent.update(Transition(S2, 1, 100.0, S0, terminated=True))  # ゴールに着いた手
    agent.end_episode()
    assert weight_row(agent, S2) == [0.0, pytest.approx(100.0), 0.0]
    assert weight_row(agent, S1) == [0.0, 0.0, pytest.approx(50.0)]
    assert weight_row(agent, S0) == [pytest.approx(25.0), 0.0, 0.0]


def test_a_rule_repeated_in_the_episode_is_reinforced_for_every_occurrence():
    agent = make_agent(discount_factor=0.5)
    agent.update(Transition(S0, 1, -1.0, S1, terminated=False))  # n=3
    agent.update(Transition(S1, 0, -1.0, S0, terminated=False))  # n=2
    agent.update(Transition(S0, 1, -1.0, S2, terminated=False))  # n=1 (n=3 と同じルール)
    agent.update(Transition(S2, 2, 10.0, S0, terminated=True))  # n=0
    agent.end_episode()
    assert weight_row(agent, S0) == [0.0, pytest.approx(10.0 * (0.5**3 + 0.5)), 0.0]
    assert weight_row(agent, S1) == [pytest.approx(2.5), 0.0, 0.0]
    assert weight_row(agent, S2) == [0.0, 0.0, pytest.approx(10.0)]


def test_discount_factor_one_gives_every_rule_on_the_route_the_full_goal_reward():
    agent = make_agent(discount_factor=1.0)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 2, 100.0, S2, terminated=True))
    agent.end_episode()
    assert weight_row(agent, S0) == [100.0, 0.0, 0.0]
    assert weight_row(agent, S1) == [0.0, 0.0, 100.0]


def test_step_rewards_are_ignored():
    def reinforced_weights(step_reward: float) -> dict:
        agent = make_agent(discount_factor=0.5)
        agent.update(Transition(S0, 0, step_reward, S1, terminated=False))
        agent.update(Transition(S1, 2, 100.0, S2, terminated=True))
        agent.end_episode()
        return agent.snapshot()["weights"]

    expected = {"0,1": [50.0, 0.0, 0.0], "1,0": [0.0, 0.0, 100.0]}
    for step_reward in (-1.0, -50.0, 7.0):
        assert reinforced_weights(step_reward) == expected


def test_truncated_episode_changes_nothing():
    agent = make_agent()
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 2, -1.0, S2, terminated=False))  # 最大手数で打ち切り
    agent.end_episode()
    assert agent.snapshot()["weights"] == {}


def test_each_episode_is_reinforced_only_from_its_own_transitions():
    agent = make_agent(discount_factor=0.5)
    for _ in range(2):
        agent.update(Transition(S0, 1, 100.0, S1, terminated=True))
        agent.end_episode()
    # 1 回目の遷移が 2 回目に持ち越されると 100 + 100 + 50 になる
    assert weight_row(agent, S0) == [0.0, pytest.approx(200.0), 0.0]


# ----- 行動選択 (重みに対する ε-greedy) -----


def test_greedy_selection_picks_the_action_with_the_largest_weight():
    agent = make_agent(epsilon_start=0.0, epsilon_min=0.0)
    set_weights(agent, S0, [0.0, 3.0, 1.0])
    assert all(agent.select_action(S0) == 1 for _ in range(20))


def test_goal_reinforcement_makes_the_goal_action_the_greedy_choice():
    agent = make_agent(epsilon_start=0.0, epsilon_min=0.0)
    agent.update(Transition(S0, 2, 100.0, S1, terminated=True))
    agent.end_episode()
    assert {agent.select_action(S0) for _ in range(50)} == {2}


def test_unreinforced_states_break_ties_randomly_while_exploring():
    agent = make_agent(epsilon_start=0.0, epsilon_min=0.0)
    assert {agent.select_action(S0) for _ in range(200)} == {0, 1, 2}


def test_epsilon_one_selects_every_action_eventually():
    agent = make_agent(epsilon_start=1.0)
    set_weights(agent, S0, [0.0, 3.0, 1.0])
    assert {agent.select_action(S0) for _ in range(200)} == {0, 1, 2}


def test_without_exploration_selection_is_greedy_and_deterministic():
    agent = make_agent(epsilon_start=1.0)
    set_weights(agent, S0, [0.0, 3.0, 3.0])
    assert {agent.select_action(S0, explore=False) for _ in range(50)} == {1}
    assert {agent.select_action((9, 9), explore=False) for _ in range(50)} == {0}  # 未強化は同値 → 番号の小さい方


def test_selection_without_exploration_changes_nothing():
    agent = make_agent()
    agent.update(Transition(S0, 1, 100.0, S1, terminated=True))
    agent.end_episode()
    before = agent.snapshot()
    state_before = agent._rng.getstate()
    for _ in range(10):
        agent.select_action(S0, explore=False)
    assert agent.snapshot() == before
    assert agent._rng.getstate() == state_before


def test_selection_does_not_add_states_to_the_weights():
    agent = make_agent()
    agent.select_action(S0)
    assert agent.snapshot()["weights"] == {}


# ----- ε の減衰 -----


def test_end_episode_decays_epsilon_down_to_the_minimum():
    agent = make_agent(epsilon_start=1.0, epsilon_min=0.3, epsilon_decay=0.5)
    seen = []
    for _ in range(4):
        agent.update(Transition(S0, 0, 100.0, S1, terminated=True))
        agent.end_episode()
        seen.append(agent.epsilon)
    assert seen == [0.5, 0.3, 0.3, 0.3]


def test_epsilon_also_decays_after_a_truncated_episode():
    agent = make_agent(epsilon_start=1.0, epsilon_decay=0.5)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.end_episode()
    assert agent.epsilon == 0.5


def test_end_episode_without_transitions_only_decays_epsilon():
    agent = make_agent(epsilon_start=1.0, epsilon_decay=0.5)
    agent.end_episode()
    assert agent.epsilon == 0.5
    assert agent.snapshot()["weights"] == {}


# ----- 記録・保存・復元 -----


def test_metrics_report_epsilon_and_weights_size():
    agent = make_agent(epsilon_start=0.8, epsilon_decay=1.0)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 1, 100.0, S2, terminated=True))
    assert agent.metrics() == {"epsilon": 0.8, "weights_size": 0.0}  # 強化前
    agent.end_episode()
    assert agent.metrics() == {"epsilon": 0.8, "weights_size": 2.0}


def test_snapshot_is_json_serializable_and_restores_to_an_equal_agent():
    agent = make_agent(discount_factor=0.5)
    agent.update(Transition(S0, 1, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 2, 100.0, S0, terminated=True))
    agent.end_episode()

    snapshot = json.loads(json.dumps(agent.snapshot()))
    assert set(snapshot) == {"epsilon", "weights"}
    restored = make_agent()
    restored.restore(snapshot)

    assert restored.snapshot() == agent.snapshot()
    assert restored.epsilon == agent.epsilon
    for state in (S0, S1):
        assert restored.select_action(state, explore=False) == agent.select_action(state, explore=False)


def test_restore_rejects_rows_of_the_wrong_length():
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.restore({"epsilon": 0.1, "weights": {"0,1": [0.0, 0.0]}})


# ----- 設定の検証 -----


@pytest.mark.parametrize(
    "kwargs",
    [
        {"discount_factor": 0.0},
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
        ProfitSharingConfig(**kwargs)


def test_config_accepts_undiscounted_credit():
    assert ProfitSharingConfig(discount_factor=1.0).discount_factor == 1.0


def test_hyperparameters_are_json_serializable_and_complete():
    config = ProfitSharingConfig(discount_factor=0.3, seed=3)
    params = ProfitSharingAgent(4, config).hyperparameters()
    assert json.loads(json.dumps(params)) == params
    assert params["discount_factor"] == 0.3
    assert params["seed"] == 3
    assert set(params) == {"discount_factor", "epsilon_start", "epsilon_min", "epsilon_decay", "seed"}


# ----- 再現性 -----


def test_same_seed_gives_same_actions():
    a, b = make_agent(seed=5), make_agent(seed=5)
    assert [a.select_action(S0) for _ in range(30)] == [b.select_action(S0) for _ in range(30)]


def test_same_seeds_give_same_training_result():
    def trained_snapshot() -> dict:
        env = LightsOutEnvironment(EnvironmentConfig(size=3, scramble_moves=2, max_steps=20, seed=4))
        agent = ProfitSharingAgent(env.n_actions, ProfitSharingConfig(seed=7))
        agent.train(env, n_episodes=100)
        return agent.snapshot()

    assert trained_snapshot() == trained_snapshot()


def test_evaluation_does_not_learn():
    env = LightsOutEnvironment(EnvironmentConfig(size=3, scramble_moves=1, max_steps=10, seed=0))
    agent = ProfitSharingAgent(env.n_actions, ProfitSharingConfig(seed=1))
    agent.train(env, n_episodes=20)
    before = agent.snapshot()
    agent.evaluate(env, n_episodes=20)
    assert agent.snapshot() == before


# ----- 実際に学習できる -----


def test_agent_learns_to_clear_boards_one_press_away():
    def make_env(seed: int) -> LightsOutEnvironment:
        return LightsOutEnvironment(
            EnvironmentConfig(size=3, scramble_moves=1, max_steps=10, seed=seed)
        )

    train_env, eval_env = make_env(0), make_env(9)
    agent = ProfitSharingAgent(
        train_env.n_actions, ProfitSharingConfig(epsilon_decay=0.99, epsilon_min=0.05, seed=1)
    )
    assert agent.evaluate(make_env(9), n_episodes=50).success_rate < 0.5

    agent.train(train_env, n_episodes=1500)

    summary = agent.evaluate(eval_env, n_episodes=50)
    assert summary.success_rate == 1.0
    # 経路全体を強化するので、最短 (1 手) とは限らず遠回りすることがある
    assert summary.mean_steps < 2.0
