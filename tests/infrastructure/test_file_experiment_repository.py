import csv
import json

from lights_out.domain.experiment import EpisodeRecord, EvaluationRecord, EvaluationSummary
from lights_out.infrastructure import FileExperimentRepository

CONFIG = {
    "settings": {"name": "demo", "n_episodes": 2},
    "environment": {"size": 4},
    "agent": {"type": "QLearningAgent", "hyperparameters": {"learning_rate": 0.1}},
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_begin_creates_a_run_directory_with_config_and_provenance(tmp_path):
    repository = FileExperimentRepository(tmp_path / "runs")
    run_id = repository.begin("demo", CONFIG)

    assert run_id.endswith("_demo")
    config = json.loads((tmp_path / "runs" / run_id / "config.json").read_text(encoding="utf-8"))
    assert config["run_id"] == run_id
    assert config["started_at"]
    assert config["settings"] == CONFIG["settings"]
    assert config["environment"] == CONFIG["environment"]
    assert config["agent"] == CONFIG["agent"]
    assert {"package_version", "python_version", "platform", "git"} <= set(config["provenance"])


def test_runs_started_in_the_same_second_get_distinct_ids(tmp_path):
    repository = FileExperimentRepository(tmp_path)
    run_ids = {repository.begin("demo", CONFIG) for _ in range(5)}
    assert len(run_ids) == 5
    assert all((tmp_path / run_id).is_dir() for run_id in run_ids)


def test_finish_writes_episodes_evaluations_agent_and_summary(tmp_path):
    repository = FileExperimentRepository(tmp_path)
    run_id = repository.begin("demo", CONFIG)
    repository.finish(
        run_id,
        episodes=[
            EpisodeRecord(1, -100.0, 100, False, {"epsilon": 0.9, "q_table_size": 10.0}),
            EpisodeRecord(2, 98.0, 3, True, {"epsilon": 0.8, "q_table_size": 12.0}),
        ],
        evaluations=[
            EvaluationRecord(0, EvaluationSummary(10, 0.0, -100.0, 100.0)),
            EvaluationRecord(2, EvaluationSummary(10, 0.5, 40.0, 50.5)),
        ],
        agent_snapshot={"epsilon": 0.8, "q_table": {"0,1": [0.0, 1.0]}},
    )
    run_dir = tmp_path / run_id

    assert read_csv(run_dir / "episodes.csv") == [
        {"episode": "1", "total_reward": "-100.0", "steps": "100", "solved": "0", "epsilon": "0.9", "q_table_size": "10.0"},
        {"episode": "2", "total_reward": "98.0", "steps": "3", "solved": "1", "epsilon": "0.8", "q_table_size": "12.0"},
    ]
    assert read_csv(run_dir / "evaluations.csv") == [
        {"episode": "0", "n_episodes": "10", "success_rate": "0.0", "mean_reward": "-100.0", "mean_steps": "100.0"},
        {"episode": "2", "n_episodes": "10", "success_rate": "0.5", "mean_reward": "40.0", "mean_steps": "50.5"},
    ]
    assert json.loads((run_dir / "agent.json").read_text(encoding="utf-8")) == {
        "epsilon": 0.8,
        "q_table": {"0,1": [0.0, 1.0]},
    }
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_episodes"] == 2
    assert summary["finished_at"]
    assert summary["final_evaluation"] == {
        "episode": 2,
        "n_episodes": 10,
        "success_rate": 0.5,
        "mean_reward": 40.0,
        "mean_steps": 50.5,
    }


def test_finish_with_no_episodes_writes_headers_only(tmp_path):
    repository = FileExperimentRepository(tmp_path)
    run_id = repository.begin("demo", CONFIG)
    repository.finish(run_id, episodes=[], evaluations=[], agent_snapshot={})

    assert read_csv(tmp_path / run_id / "episodes.csv") == []
    header = (tmp_path / run_id / "episodes.csv").read_text(encoding="utf-8").strip()
    assert header == "episode,total_reward,steps,solved"
    summary = json.loads((tmp_path / run_id / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_evaluation"] is None
