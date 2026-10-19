import pytest

from lights_out.application import ExperimentRunner, ExperimentSettings
from lights_out.domain.agent import Agent, QLearningAgent, QLearningConfig
from lights_out.domain.environment import EnvironmentConfig, LightsOutEnvironment
from lights_out.domain.experiment import ExperimentRepository


class InMemoryRepository(ExperimentRepository):
    def __init__(self) -> None:
        self.begun: list[tuple[str, dict]] = []
        self.finished: dict | None = None

    def begin(self, name, config):
        self.begun.append((name, config))
        return f"run-{len(self.begun)}"

    def finish(self, run_id, *, episodes, evaluations, agent_snapshot):
        self.finished = {
            "run_id": run_id,
            "episodes": episodes,
            "evaluations": evaluations,
            "agent_snapshot": agent_snapshot,
        }


class AlwaysPressFirstCell(Agent):
    """学習せず、常にマス 0 を押すエージェント。結果が開始盤面だけで決まる。"""

    def select_action(self, state, explore=True):
        return 0

    def update(self, transition):
        pass

    def hyperparameters(self):
        return {}

    def snapshot(self):
        return {}

    def restore(self, snapshot):
        pass


def make_env(**kwargs) -> LightsOutEnvironment:
    kwargs = {"size": 3, "scramble_moves": 1, "max_steps": 10, "seed": 0, **kwargs}
    return LightsOutEnvironment(EnvironmentConfig(**kwargs))


def make_agent(env: LightsOutEnvironment, **kwargs) -> QLearningAgent:
    kwargs = {"epsilon_decay": 0.99, "seed": 1, **kwargs}
    return QLearningAgent(env.n_actions, QLearningConfig(**kwargs))


def run(settings: ExperimentSettings, repository=None, **agent_kwargs):
    repository = repository or InMemoryRepository()
    env = make_env()
    agent = make_agent(env, **agent_kwargs)
    result = ExperimentRunner(repository).run(env, agent, settings)
    return result, repository, agent


def test_run_trains_the_requested_number_of_episodes():
    result, _, _ = run(ExperimentSettings(n_episodes=50, eval_interval=0, eval_episodes=5))
    assert [r.episode for r in result.episodes] == list(range(1, 51))


def test_run_evaluates_before_training_at_each_interval_and_at_the_end():
    result, _, _ = run(ExperimentSettings(n_episodes=50, eval_interval=20, eval_episodes=5))
    assert [e.episode for e in result.evaluations] == [0, 20, 40, 50]


def test_run_does_not_evaluate_twice_when_the_last_episode_is_on_an_interval():
    result, _, _ = run(ExperimentSettings(n_episodes=40, eval_interval=20, eval_episodes=5))
    assert [e.episode for e in result.evaluations] == [0, 20, 40]


def test_run_without_interval_evaluates_only_before_and_after():
    result, _, _ = run(ExperimentSettings(n_episodes=30, eval_interval=0, eval_episodes=5))
    assert [e.episode for e in result.evaluations] == [0, 30]


def test_run_with_zero_episodes_evaluates_once():
    result, _, _ = run(ExperimentSettings(n_episodes=0, eval_interval=10, eval_episodes=5))
    assert [e.episode for e in result.evaluations] == [0]
    assert result.episodes == []


def test_evaluation_uses_the_requested_number_of_episodes():
    result, _, _ = run(ExperimentSettings(n_episodes=10, eval_interval=0, eval_episodes=7))
    assert all(e.summary.n_episodes == 7 for e in result.evaluations)


def test_run_saves_config_before_training_and_results_after():
    settings = ExperimentSettings(name="demo", n_episodes=30, eval_interval=0, eval_episodes=5, eval_seed=3)
    result, repository, agent = run(settings)

    (name, config), = repository.begun
    assert name == "demo"
    assert config["settings"] == {
        "name": "demo",
        "n_episodes": 30,
        "eval_interval": 0,
        "eval_episodes": 5,
        "eval_seed": 3,
    }
    assert config["environment"] == {
        "size": 3,
        "max_steps": 10,
        "scramble_moves": 1,
        "step_reward": -1.0,
        "goal_reward": 100.0,
        "seed": 0,
    }
    assert config["agent"]["type"] == "QLearningAgent"
    assert config["agent"]["hyperparameters"] == agent.hyperparameters()

    assert result.run_id == "run-1"
    assert repository.finished["run_id"] == "run-1"
    assert repository.finished["episodes"] == result.episodes
    assert repository.finished["evaluations"] == result.evaluations
    assert repository.finished["agent_snapshot"] == agent.snapshot()


def test_evaluation_uses_the_same_start_boards_every_time():
    # 方策が変わらないエージェントなら、同じ盤面で評価している限り評価結果は毎回同じになる
    env = make_env()
    settings = ExperimentSettings(n_episodes=20, eval_interval=5, eval_episodes=30)
    result = ExperimentRunner(InMemoryRepository()).run(env, AlwaysPressFirstCell(env.n_actions), settings)

    summaries = [e.summary for e in result.evaluations]
    assert len(summaries) == 5
    assert all(s == summaries[0] for s in summaries)
    assert 0.0 < summaries[0].success_rate < 1.0  # 盤面によって結果が分かれる = 判定として意味がある


def test_evaluation_does_not_affect_training():
    # 評価は学習用の環境・エージェントの乱数を使わないので、評価の頻度を変えても学習結果は同じ
    sparse, _, sparse_agent = run(ExperimentSettings(n_episodes=60, eval_interval=0, eval_episodes=5))
    dense, _, dense_agent = run(ExperimentSettings(n_episodes=60, eval_interval=5, eval_episodes=20))
    assert sparse_agent.snapshot() == dense_agent.snapshot()
    assert sparse.episodes == dense.episodes


def test_same_settings_reproduce_the_same_result():
    settings = ExperimentSettings(n_episodes=60, eval_interval=20, eval_episodes=10)
    first, _, _ = run(settings)
    second, _, _ = run(settings)
    assert first.episodes == second.episodes
    assert first.evaluations == second.evaluations


def test_run_does_not_touch_the_training_environment_config():
    env = make_env()
    agent = make_agent(env)
    ExperimentRunner(InMemoryRepository()).run(
        env, agent, ExperimentSettings(n_episodes=5, eval_interval=0, eval_episodes=3, eval_seed=99)
    )
    assert env.config.seed == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"n_episodes": -1}, {"eval_interval": -1}, {"eval_episodes": 0}],
)
def test_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ExperimentSettings(**kwargs)
