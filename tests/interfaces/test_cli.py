import json

import pytest

from lights_out.interfaces.cli import main

SMALL_RUN = ["--size", "3", "--scramble-moves", "1", "--max-steps", "10", "--eval-episodes", "5"]


def only_run_dir(runs_dir):
    (run_dir,) = runs_dir.iterdir()
    return run_dir


def test_main_trains_and_records_the_run(tmp_path, capsys):
    exit_code = main([*SMALL_RUN, "--episodes", "50", "--eval-interval", "25", "--runs-dir", str(tmp_path)])

    assert exit_code == 0
    run_dir = only_run_dir(tmp_path)
    assert run_dir.name in capsys.readouterr().out
    for name in ("config.json", "episodes.csv", "evaluations.csv", "agent.json", "summary.json"):
        assert (run_dir / name).is_file()
    assert len((run_dir / "episodes.csv").read_text(encoding="utf-8").splitlines()) == 51


def test_main_records_the_given_options_in_config(tmp_path):
    main(
        [
            *SMALL_RUN,
            "--episodes", "10",
            "--seed", "7",
            "--learning-rate", "0.3",
            "--epsilon-decay", "0.9",
            "--name", "trial",
            "--runs-dir", str(tmp_path),
        ]
    )
    config = json.loads((only_run_dir(tmp_path) / "config.json").read_text(encoding="utf-8"))

    assert config["settings"]["name"] == "trial"
    assert config["settings"]["n_episodes"] == 10
    assert config["environment"]["size"] == 3
    assert config["environment"]["seed"] == 7
    assert config["agent"]["hyperparameters"]["learning_rate"] == 0.3
    assert config["agent"]["hyperparameters"]["epsilon_decay"] == 0.9
    assert config["agent"]["hyperparameters"]["seed"] == 8
    assert config["settings"]["eval_seed"] == 9


def test_main_is_reproducible_for_the_same_seed(tmp_path):
    for name in ("a", "b"):
        main([*SMALL_RUN, "--episodes", "40", "--seed", "3", "--name", name, "--runs-dir", str(tmp_path)])
    run_a, run_b = sorted(tmp_path.iterdir(), key=lambda p: p.name.split("_")[1])

    assert (run_a / "episodes.csv").read_text(encoding="utf-8") == (run_b / "episodes.csv").read_text(encoding="utf-8")
    assert (run_a / "agent.json").read_text(encoding="utf-8") == (run_b / "agent.json").read_text(encoding="utf-8")


def test_demo_plays_the_learned_policy_until_solved(tmp_path, capsys):
    main([*SMALL_RUN, "--episodes", "300", "--runs-dir", str(tmp_path), "--demo"])
    out = capsys.readouterr().out
    assert "step 0/10  solved: False" in out
    assert "を押す" in out
    assert out.rstrip().splitlines()[-4].endswith("solved: True")  # 3x3 の盤面 3 行の前の行


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--size", "0"],
        ["--scramble-moves", "10", "--size", "3"],
        ["--learning-rate", "0"],
        ["--episodes", "-1"],
        ["--name", "a/b"],
    ],
)
def test_invalid_options_exit_with_a_usage_error_and_write_nothing(tmp_path, capsys, bad_args):
    with pytest.raises(SystemExit) as exit_info:
        main([*bad_args, "--runs-dir", str(tmp_path)])
    assert exit_info.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
