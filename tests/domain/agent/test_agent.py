import pytest

from lights_out.domain.agent import Agent, Transition
from lights_out.domain.environment import EnvironmentConfig, LightsOutEnvironment

# 3x3 で左上 3 マスが点灯。マス 0 を押すと全消灯になる。
ONE_PRESS_AWAY = [1, 1, 0, 1, 0, 0, 0, 0, 0]


class ScriptedAgent(Agent):
    """決められた順に行動し、呼ばれ方を記録するだけのエージェント。"""

    def __init__(self, actions: list[int], n_actions: int = 9) -> None:
        super().__init__(n_actions)
        self._actions = actions
        self._index = 0
        self.transitions: list[Transition] = []
        self.explore_flags: list[bool] = []
        self.episodes_ended = 0

    def select_action(self, state, explore=True):
        self.explore_flags.append(explore)
        action = self._actions[self._index % len(self._actions)]
        self._index += 1
        return action

    def update(self, transition):
        self.transitions.append(transition)

    def end_episode(self):
        self.episodes_ended += 1

    def metrics(self):
        return {"episodes_ended": float(self.episodes_ended)}

    def hyperparameters(self):
        return {}

    def snapshot(self):
        return {}

    def restore(self, snapshot):
        pass


class FixedStartEnvironment(LightsOutEnvironment):
    """reset のたびに同じ盤面から始める (エージェントの処理だけを確かめる用)。"""

    def reset(self, initial_state=None):
        return super().reset(initial_state or ONE_PRESS_AWAY)


def make_env(max_steps=100) -> FixedStartEnvironment:
    return FixedStartEnvironment(EnvironmentConfig(size=3, max_steps=max_steps, seed=0))


def test_agent_is_abstract():
    with pytest.raises(TypeError):
        Agent(n_actions=4)  # type: ignore[abstract]


def test_n_actions_must_be_positive():
    with pytest.raises(ValueError):
        ScriptedAgent([0], n_actions=0)


@pytest.mark.parametrize("n_actions", [4, 16])
def test_agent_rejects_environment_with_different_number_of_actions(n_actions):
    agent = ScriptedAgent([0], n_actions=n_actions)  # 環境は 3x3 = 9 行動
    with pytest.raises(ValueError):
        agent.play_episode(make_env(), learn=True)
    with pytest.raises(ValueError):
        agent.train(make_env(), n_episodes=1)
    with pytest.raises(ValueError):
        agent.evaluate(make_env(), n_episodes=1)
    assert agent.transitions == []


def test_play_episode_learning_passes_every_transition_to_update():
    agent = ScriptedAgent([8, 8, 0])  # 8 を 2 回押して元に戻し、0 で全消灯
    record = agent.play_episode(make_env(), learn=True, episode=7)

    assert [t.action for t in agent.transitions] == [8, 8, 0]
    first, second, last = agent.transitions
    assert first.state == ONE_PRESS_AWAY
    assert first.next_state == [1, 1, 0, 1, 0, 1, 0, 1, 1]
    assert first.reward == -1.0
    assert not first.terminated
    assert second.state == first.next_state
    assert second.next_state == ONE_PRESS_AWAY
    assert last.state == ONE_PRESS_AWAY
    assert last.next_state == [0] * 9
    assert last.terminated
    assert last.reward == 100.0

    assert record.episode == 7
    assert record.steps == 3
    assert record.solved
    assert record.total_reward == 98.0


def test_play_episode_learning_explores_and_ends_the_episode_once():
    agent = ScriptedAgent([0])
    agent.play_episode(make_env(), learn=True)
    assert agent.explore_flags == [True]
    assert agent.episodes_ended == 1


def test_play_episode_without_learning_neither_updates_nor_explores():
    agent = ScriptedAgent([0])
    record = agent.play_episode(make_env(), learn=False)
    assert agent.transitions == []
    assert agent.explore_flags == [False]
    assert agent.episodes_ended == 0
    assert record.solved


def test_record_metrics_come_from_the_agent_after_the_episode():
    agent = ScriptedAgent([0])
    record = agent.play_episode(make_env(), learn=True)
    assert record.metrics == {"episodes_ended": 1.0}


def test_truncated_episode_is_unsolved_and_its_last_transition_is_not_terminated():
    agent = ScriptedAgent([8])  # 同じマスを押し続けて解けない
    record = agent.play_episode(make_env(max_steps=3), learn=True)
    assert record.steps == 3
    assert not record.solved
    assert record.total_reward == -3.0
    assert not agent.transitions[-1].terminated


def test_train_runs_the_requested_number_of_episodes():
    agent = ScriptedAgent([0])
    records = agent.train(make_env(), n_episodes=5)
    assert [r.episode for r in records] == [1, 2, 3, 4, 5]
    assert agent.episodes_ended == 5


def test_train_calls_callback_after_each_episode():
    agent = ScriptedAgent([0])
    seen = []
    records = agent.train(make_env(), n_episodes=3, on_episode_end=seen.append)
    assert seen == records


def test_train_with_zero_episodes_does_nothing():
    agent = ScriptedAgent([0])
    assert agent.train(make_env(), n_episodes=0) == []


def test_train_rejects_negative_episodes():
    with pytest.raises(ValueError):
        ScriptedAgent([0]).train(make_env(), n_episodes=-1)


def test_evaluate_summarizes_without_learning():
    agent = ScriptedAgent([0])
    summary = agent.evaluate(make_env(), n_episodes=4)
    assert summary.n_episodes == 4
    assert summary.success_rate == 1.0
    assert summary.mean_steps == 1.0
    assert summary.mean_reward == 100.0
    assert agent.transitions == []
    assert agent.episodes_ended == 0


def test_evaluate_reports_partial_success():
    agent = ScriptedAgent([0, 8])  # 1 回目は解け、2 回目は 8 を押し続けて打ち切り
    summary = agent.evaluate(make_env(max_steps=2), n_episodes=2)
    assert summary.success_rate == 0.5
    assert summary.mean_steps == 1.5


def test_evaluate_requires_at_least_one_episode():
    with pytest.raises(ValueError):
        ScriptedAgent([0]).evaluate(make_env(), n_episodes=0)
