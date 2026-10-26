import json
from dataclasses import asdict

import pytest

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
from lights_out.interfaces.cli import METHODS, main

SMALL_RUN = ["--size", "3", "--scramble-moves", "1", "--max-steps", "10", "--eval-episodes", "5"]

# --method の名前 -> (エージェント, 設定, 学習結果の表のキー)
EXPECTED = {
    "q-learning": (QLearningAgent, QLearningConfig, "q_table"),
    "sarsa": (SarsaAgent, SarsaConfig, "q_table"),
    "profit-sharing": (ProfitSharingAgent, ProfitSharingConfig, "weights"),
    "monte-carlo": (MonteCarloAgent, MonteCarloConfig, "q_table"),
}
METHOD_NAMES = list(EXPECTED)


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


# ----- --method -----


def read_config(run_dir):
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def test_registry_has_exactly_the_tested_methods():
    # 手法を足したら、EXPECTED (= この下のテスト全部) にも足すこと
    assert {name: (agent, config) for name, (agent, config, _) in EXPECTED.items()} == METHODS


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_each_method_trains_and_records_the_run(tmp_path, method):
    agent_cls, config_cls, table_key = EXPECTED[method]

    exit_code = main([*SMALL_RUN, "--method", method, "--episodes", "30", "--runs-dir", str(tmp_path)])

    assert exit_code == 0
    run_dir = only_run_dir(tmp_path)
    assert run_dir.name.endswith(f"_{method}")  # --name を省略すると手法の名前になる
    config = read_config(run_dir)
    assert config["settings"]["name"] == method
    assert config["agent"]["type"] == agent_cls.__name__
    assert config["agent"]["hyperparameters"] == asdict(config_cls(seed=1))  # その手法自身の既定値 + seed+1
    assert (run_dir / "summary.json").is_file()
    assert table_key in json.loads((run_dir / "agent.json").read_text(encoding="utf-8"))
    header = (run_dir / "episodes.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert f"{table_key}_size" in header  # q_table_size / weights_size


def test_default_method_is_q_learning(tmp_path):
    main([*SMALL_RUN, "--episodes", "10", "--runs-dir", str(tmp_path)])
    run_dir = only_run_dir(tmp_path)

    assert read_config(run_dir)["agent"]["type"] == "QLearningAgent"
    assert run_dir.name.endswith("_q-learning")


def test_unknown_method_exits_with_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--method", "dqn", "--runs-dir", str(tmp_path)])

    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "--method" in err
    assert "Traceback" not in err
    assert list(tmp_path.iterdir()) == []


def test_explicit_name_wins_over_the_method_name(tmp_path):
    main([*SMALL_RUN, "--method", "sarsa", "--episodes", "5", "--name", "trial", "--runs-dir", str(tmp_path)])

    assert only_run_dir(tmp_path).name.endswith("_trial")


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_hyperparameter_options_reach_the_chosen_methods_config(tmp_path, method):
    main(
        [
            *SMALL_RUN,
            "--method", method,
            "--episodes", "5",
            "--seed", "4",
            "--discount-factor", "0.2",
            "--epsilon-start", "0.8",
            "--epsilon-min", "0.02",
            "--epsilon-decay", "0.9",
            "--runs-dir", str(tmp_path),
        ]
    )
    hyperparameters = read_config(only_run_dir(tmp_path))["agent"]["hyperparameters"]

    assert hyperparameters["discount_factor"] == 0.2
    assert hyperparameters["epsilon_start"] == 0.8
    assert hyperparameters["epsilon_min"] == 0.02
    assert hyperparameters["epsilon_decay"] == 0.9
    assert hyperparameters["seed"] == 5


@pytest.mark.parametrize("method", [m for m in METHOD_NAMES if m != "profit-sharing"])
def test_learning_rate_option_reaches_the_methods_with_a_learning_rate(tmp_path, method):
    main([*SMALL_RUN, "--method", method, "--episodes", "5", "--learning-rate", "0.3", "--runs-dir", str(tmp_path)])

    assert read_config(only_run_dir(tmp_path))["agent"]["hyperparameters"]["learning_rate"] == 0.3


def test_an_option_the_method_does_not_have_is_a_usage_error_and_writes_nothing(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([*SMALL_RUN, "--method", "profit-sharing", "--learning-rate", "0.3", "--runs-dir", str(tmp_path)])

    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "--learning-rate は profit-sharing では指定できません" in err  # usage にも載る語でなく、エラー文そのもの
    assert "Traceback" not in err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--method", "profit-sharing", "--discount-factor", "0"],
        ["--method", "sarsa", "--epsilon-min", "2"],
        ["--method", "monte-carlo", "--learning-rate", "0"],
        ["--method", "sarsa", "--name", "a/b"],
    ],
)
def test_invalid_values_for_other_methods_are_usage_errors_too(tmp_path, capsys, bad_args):
    with pytest.raises(SystemExit) as exit_info:
        main([*bad_args, "--runs-dir", str(tmp_path)])

    assert exit_info.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_each_method_is_reproducible_for_the_same_seed(tmp_path, method):
    for name in ("a", "b"):
        main([*SMALL_RUN, "--method", method, "--episodes", "40", "--seed", "3", "--name", name,
              "--runs-dir", str(tmp_path)])
    run_a, run_b = sorted(tmp_path.iterdir(), key=lambda p: p.name.split("_")[1])

    assert (run_a / "episodes.csv").read_text(encoding="utf-8") == (run_b / "episodes.csv").read_text(encoding="utf-8")
    assert (run_a / "agent.json").read_text(encoding="utf-8") == (run_b / "agent.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_demo_works_for_each_method(tmp_path, capsys, method):
    main([*SMALL_RUN, "--method", method, "--episodes", "300", "--runs-dir", str(tmp_path), "--demo"])
    out = capsys.readouterr().out

    assert "step 0/10  solved: False" in out
    assert out.count("を押す") == 1  # 探索せず学習した方策どおりに動けば、1 手で解ける盤面を 1 手で解く
    assert out.rstrip().splitlines()[-4].endswith("solved: True")  # 3x3 の盤面 3 行の前の行


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_saved_agent_json_can_be_restored_into_a_fresh_agent(tmp_path, method):
    agent_cls, _, table_key = EXPECTED[method]
    main([*SMALL_RUN, "--method", method, "--episodes", "100", "--runs-dir", str(tmp_path)])
    saved = json.loads((only_run_dir(tmp_path) / "agent.json").read_text(encoding="utf-8"))

    fresh = agent_cls(n_actions=9)  # 3x3 の盤面は 9 マス
    fresh.restore(saved)

    assert fresh.snapshot() == saved
    assert saved[table_key]  # 訪れた盤面がある
    key, row = next(iter(saved[table_key].items()))
    state = tuple(int(cell) for cell in key.split(","))
    assert fresh.select_action(state, explore=False) == row.index(max(row))


def test_help_lists_the_methods_and_where_their_defaults_differ(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    out = capsys.readouterr().out

    text = " ".join(out.split())  # argparse の折り返しを無視して比べる
    ps, ql = ProfitSharingConfig().discount_factor, QLearningConfig().discount_factor

    assert exit_info.value.code == 0
    for method in METHOD_NAMES:
        assert method in text
    assert f"q-learning/sarsa/monte-carlo {ql}, profit-sharing {ps}" in text  # 既定値は config クラスから引いている
    assert "profit-sharing では指定不可" in text  # --learning-rate が profit-sharing にないこと
