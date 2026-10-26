import json

import pytest

from lights_out.domain.agent import SarsaAgent, SarsaConfig, Transition
from lights_out.domain.environment import EnvironmentConfig, LightsOutEnvironment

S0 = (0, 1)
S1 = (1, 0)


def make_agent(**kwargs) -> SarsaAgent:
    kwargs.setdefault("seed", 0)
    return SarsaAgent(n_actions=3, config=SarsaConfig(**kwargs))


def q_row(agent: SarsaAgent, state) -> list[float]:
    return agent.snapshot()["q_table"][",".join(map(str, state))]


def set_q(agent: SarsaAgent, state, values: list[float]) -> None:
    snapshot = agent.snapshot()
    snapshot["q_table"][",".join(map(str, state))] = values
    agent.restore(snapshot)


def truncated_last_targets(**kwargs) -> set[float]:
    """S1 の価値が [8, 0, 0] のとき、打ち切りで終わった 1 手 (S0, 行動 0, 報酬 -1) で更新された Q(S0, 0) の集合。

    最後の a' は乱数で決まるので、シードを変えて 30 回試す (α=1, γ=1 なので Q = -1 + Q(S1, a'))。
    """
    results = set()
    for seed in range(30):
        agent = make_agent(learning_rate=1.0, discount_factor=1.0, seed=seed, **kwargs)
        set_q(agent, S1, [8.0, 0.0, 0.0])
        agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
        agent.end_episode()
        results.add(q_row(agent, S0)[0])
    return results


# ----- 更新式 -----


def test_update_uses_the_next_action_actually_taken_not_the_max():
    agent = make_agent(learning_rate=0.5, discount_factor=0.9)
    set_q(agent, S1, [2.0, 4.0, 1.0])
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 2, -1.0, S0, terminated=False))  # S1 では最大でない行動 2 を選んだ
    # target = -1 + 0.9 * Q(S1, 2) = -0.1 (Q 学習なら max の 4 を使って 2.6)
    assert q_row(agent, S0) == [pytest.approx(-0.05), 0.0, 0.0]


def test_update_never_bootstraps_from_a_terminated_transition_even_if_more_follow():
    agent = make_agent(learning_rate=0.5)
    set_q(agent, S1, [100.0, 100.0, 100.0])
    agent.update(Transition(S0, 0, 10.0, S1, terminated=True))
    agent.update(Transition(S1, 1, -1.0, S0, terminated=False))
    assert q_row(agent, S0) == [pytest.approx(5.0), 0.0, 0.0]


def test_episode_is_learned_in_order_each_step_using_the_next_action_taken():
    agent = make_agent(learning_rate=0.5, discount_factor=0.9)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 1, -1.0, S0, terminated=False))
    agent.update(Transition(S0, 0, 10.0, S1, terminated=True))
    # 1 手目: Q(S0,0) = 0.5 * (-1 + 0.9 * Q(S1,1)=0)                      = -0.5
    # 2 手目: Q(S1,1) = 0.5 * (-1 + 0.9 * Q(S0,0)=-0.5)                   = -0.725 (1 手目の更新後の値を使う)
    assert q_row(agent, S0) == [pytest.approx(-0.5), 0.0, 0.0]
    assert q_row(agent, S1) == [0.0, pytest.approx(-0.725), 0.0]
    agent.end_episode()
    # 3 手目 (クリア): Q(S0,0) = -0.5 + 0.5 * (10 - (-0.5))               = 4.75
    assert q_row(agent, S0) == [pytest.approx(4.75), 0.0, 0.0]
    assert q_row(agent, S1) == [0.0, pytest.approx(-0.725), 0.0]


def test_update_accepts_list_states():
    agent = make_agent(learning_rate=1.0)
    agent.update(Transition([0, 1], 0, -1.0, [1, 0], terminated=False))
    agent.update(Transition([1, 0], 1, -1.0, [0, 1], terminated=True))
    agent.end_episode()
    assert q_row(agent, S0)[0] == -1.0


def test_initial_q_is_used_for_unvisited_states():
    agent = make_agent(learning_rate=1.0, discount_factor=1.0, initial_q=5.0)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 1, -1.0, S0, terminated=True))
    assert q_row(agent, S0) == [4.0, 5.0, 5.0]  # -1 + 1 * Q(S1, 1)=初期値 5


def test_end_episode_completes_the_last_transition_with_the_reward_when_terminated():
    agent = make_agent(learning_rate=0.5, discount_factor=0.9)
    set_q(agent, S0, [100.0, 100.0, 100.0])
    agent.update(Transition(S0, 1, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 2, 10.0, S0, terminated=True))
    agent.end_episode()
    # 最後の手はクリアなので S0 の価値 (100) を使わず target = r
    assert q_row(agent, S1) == [0.0, 0.0, pytest.approx(5.0)]


def test_end_episode_bootstraps_when_the_episode_was_only_truncated():
    # 打ち切りは Transition.terminated=False で渡る。S1 での次の行動 (ε=0 なので最大の 0) の価値を使う
    agent = make_agent(learning_rate=1.0, discount_factor=0.5, epsilon_start=0.0, epsilon_min=0.0)
    set_q(agent, S1, [8.0, 0.0, 0.0])
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.end_episode()
    assert q_row(agent, S0)[0] == pytest.approx(3.0)


def test_bootstrapping_from_an_unvisited_next_state_does_not_add_it_to_the_table():
    agent = make_agent()
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.end_episode()
    assert set(agent.snapshot()["q_table"]) == {"0,1"}


def test_truncated_last_transition_samples_the_next_action_from_the_behaviour_policy():
    # ε=1 なら a' は一様ランダム。最大の価値 (8) を使う場合 (7) と、そうでない場合 (-1) の両方が現れる
    assert truncated_last_targets(epsilon_start=1.0, epsilon_decay=1.0) == {7.0, -1.0}


# ----- ε の減衰・エピソードの区切り -----


def test_end_episode_decays_epsilon_down_to_the_minimum():
    agent = make_agent(epsilon_start=1.0, epsilon_min=0.3, epsilon_decay=0.5)
    seen = []
    for _ in range(4):
        agent.end_episode()
        seen.append(agent.epsilon)
    assert seen == [0.5, 0.3, 0.3, 0.3]


def test_epsilon_decays_only_after_the_last_transition_is_completed():
    # 減衰が先なら ε≈0 になって a' は必ず最大 (7)。更新が先なら ε=1 のままなので -1 になる場合もある
    assert truncated_last_targets(epsilon_start=1.0, epsilon_min=0.0, epsilon_decay=1e-9) == {7.0, -1.0}


def test_first_update_of_the_next_episode_does_not_touch_the_previous_episode():
    agent = make_agent(learning_rate=0.5, epsilon_start=0.0, epsilon_min=0.0)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.end_episode()
    before = q_row(agent, S0).copy()

    agent.update(Transition(S1, 2, -1.0, S0, terminated=False))  # 次のエピソードの 1 手目

    assert q_row(agent, S0) == before


def test_end_episode_completes_the_last_transition_only_once():
    agent = make_agent(learning_rate=0.5)
    agent.update(Transition(S0, 1, 10.0, S1, terminated=True))
    agent.end_episode()
    agent.end_episode()
    assert q_row(agent, S0) == [0.0, pytest.approx(5.0), 0.0]


# ----- 環境を使ったエピソード -----


class RecordingSarsaAgent(SarsaAgent):
    """update に渡された遷移を記録する SARSA (教科書どおりの計算と突き合わせる用)。"""

    def __init__(self, n_actions: int, config: SarsaConfig) -> None:
        super().__init__(n_actions, config)
        self.transitions: list[Transition] = []

    def update(self, transition: Transition) -> None:
        self.transitions.append(transition)
        super().update(transition)


def apply_textbook_sarsa(
    table: dict, config: SarsaConfig, n_actions: int, episode: list[Transition]
) -> None:
    """教科書どおりの SARSA で、1 エピソード分の遷移を順に表 (snapshot と同じ形) へ適用する。

    打ち切りで終わったエピソードは、最後の次の行動を最大の価値の行動 (ε=0 の方策) とみなす。
    """

    def row(state) -> list[float]:
        return table.setdefault(",".join(map(str, state)), [config.initial_q] * n_actions)

    for step, transition in enumerate(episode):
        target = transition.reward
        if step + 1 < len(episode):
            following = episode[step + 1]
            target += config.discount_factor * row(following.state)[following.action]
        elif not transition.terminated:
            target += config.discount_factor * max(row(transition.next_state))
        values = row(transition.state)
        values[transition.action] += config.learning_rate * (target - values[transition.action])


def assert_tables_match(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key, values in expected.items():
        assert actual[key] == pytest.approx(values)


def test_episodes_on_a_real_environment_follow_the_sarsa_rule_at_every_step():
    env = LightsOutEnvironment(EnvironmentConfig(size=2, scramble_moves=2, max_steps=500, seed=3))
    config = SarsaConfig(epsilon_start=0.5, epsilon_decay=0.9, seed=1)
    agent = RecordingSarsaAgent(env.n_actions, config)
    expected: dict = {}
    for episode in range(1, 6):
        start = len(agent.transitions)
        record = agent.play_episode(env, learn=True, episode=episode)
        assert record.solved
        apply_textbook_sarsa(expected, config, env.n_actions, agent.transitions[start:])
        assert_tables_match(agent.snapshot()["q_table"], expected)
    assert len(agent.transitions) > 5  # 複数手のエピソードを含む


def test_truncated_episode_on_a_real_environment_bootstraps_the_last_step():
    env = LightsOutEnvironment(EnvironmentConfig(size=4, scramble_moves=8, max_steps=3, seed=3))
    # initial_q を 0 以外にして、未訪問の盤面でも先の価値が 0 にならないようにする
    config = SarsaConfig(epsilon_start=0.0, epsilon_min=0.0, initial_q=5.0, seed=1)
    agent = RecordingSarsaAgent(env.n_actions, config)
    record = agent.play_episode(env, learn=True)
    assert not record.solved
    assert len(agent.transitions) == 3
    expected: dict = {}
    apply_textbook_sarsa(expected, config, env.n_actions, agent.transitions)
    assert_tables_match(agent.snapshot()["q_table"], expected)


# ----- 行動選択 -----


def test_selection_without_exploration_changes_nothing():
    agent = make_agent()
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    before = agent.snapshot()
    rng_before = agent._rng.getstate()
    for _ in range(10):
        agent.select_action(S0, explore=False)
    assert agent.snapshot() == before
    assert agent._rng.getstate() == rng_before


# ----- 記録 -----


def test_metrics_report_epsilon_and_table_size():
    agent = make_agent(epsilon_start=0.8, epsilon_decay=1.0)
    agent.update(Transition(S0, 0, -1.0, S1, terminated=False))
    agent.update(Transition(S1, 1, 10.0, S0, terminated=True))
    agent.end_episode()
    assert agent.metrics() == {"epsilon": 0.8, "q_table_size": 2.0}


def test_hyperparameters_are_json_serializable_and_complete():
    config = SarsaConfig(learning_rate=0.2, seed=3)
    params = SarsaAgent(4, config).hyperparameters()
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


def test_same_seed_gives_same_learning():
    def train(seed: int) -> dict:
        env = LightsOutEnvironment(EnvironmentConfig(size=2, scramble_moves=2, max_steps=500, seed=3))
        agent = SarsaAgent(env.n_actions, SarsaConfig(seed=seed))
        agent.train(env, n_episodes=20)
        return agent.snapshot()

    assert train(5) == train(5)
    assert train(5) != train(6)


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
        SarsaConfig(**kwargs)


# ----- 実際に学習できる -----


def test_agent_learns_to_clear_boards_one_press_away():
    def make_env(seed: int) -> LightsOutEnvironment:
        return LightsOutEnvironment(
            EnvironmentConfig(size=3, scramble_moves=1, max_steps=10, seed=seed)
        )

    train_env, eval_env = make_env(0), make_env(9)
    agent = SarsaAgent(train_env.n_actions, SarsaConfig(epsilon_decay=0.99, epsilon_min=0.05, seed=1))
    assert agent.evaluate(make_env(9), n_episodes=50).success_rate < 0.5

    agent.train(train_env, n_episodes=1500)

    summary = agent.evaluate(eval_env, n_episodes=50)
    assert summary.success_rate == 1.0
    assert summary.mean_steps == 1.0
