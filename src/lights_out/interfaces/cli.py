"""コマンドライン: 表形式の強化学習エージェントでライツアウトを学習する。

    python -m lights_out --episodes 10000 --seed 0
    python -m lights_out --method sarsa --episodes 10000 --seed 0

手法は ``--method`` で選ぶ (q-learning / sarsa / profit-sharing / monte-carlo)。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import Any

from lights_out.application import ExperimentRunner, ExperimentSettings
from lights_out.domain.agent import (
    Agent,
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

#: --method の名前 -> (エージェント, 設定)。手法を足すときはここに 1 行足す。
METHODS: dict[str, tuple[Callable[..., Agent], type]] = {
    "q-learning": (QLearningAgent, QLearningConfig),
    "sarsa": (SarsaAgent, SarsaConfig),
    "profit-sharing": (ProfitSharingAgent, ProfitSharingConfig),
    "monte-carlo": (MonteCarloAgent, MonteCarloConfig),
}
DEFAULT_METHOD = "q-learning"

#: 手法の設定クラスに渡せるハイパーパラメータ (オプション名は ``--`` + 下線をハイフンにしたもの) と、help の説明。
HYPERPARAMETERS = {
    "learning_rate": "学習率 α",
    "discount_factor": "割引率 γ (profit-sharing では信用割当の減衰率)",
    "epsilon_start": "探索率 ε の初期値",
    "epsilon_min": "ε の下限",
    "epsilon_decay": "エピソードごとに ε に掛ける係数",
}


def _option(name: str) -> str:
    return "--" + name.replace("_", "-")


def _default_help(name: str) -> str:
    """ハイパーパラメータ name の既定値を手法ごとに示す短い文。設定クラスから読むので、既定値を変えても食い違わない。"""
    defaults: dict[Any, list[str]] = {}
    unsupported = []
    for method, (_, config_cls) in METHODS.items():
        field = next((f for f in fields(config_cls) if f.name == name), None)
        if field is None:
            unsupported.append(method)
        else:
            defaults.setdefault(field.default, []).append(method)
    if len(defaults) == 1:
        text = f"既定: {next(iter(defaults))}"
    else:
        text = "既定: " + ", ".join(f"{'/'.join(methods)} {value}" for value, methods in defaults.items())
    if unsupported:
        text += f" ({'/'.join(unsupported)} では指定不可)"
    return text


def build_parser() -> argparse.ArgumentParser:
    env_defaults = EnvironmentConfig()
    settings_defaults = ExperimentSettings()

    parser = argparse.ArgumentParser(
        prog="lights-out",
        description="ライツアウトを表形式の強化学習 (--method で手法を選ぶ) で学習し、結果を runs/ に記録する",
    )
    parser.add_argument("--seed", type=int, default=0, help="基準シード (環境=seed, エージェント=seed+1, 評価=seed+2)")
    parser.add_argument("--name", default=None, help="run の名前 (既定: --method の名前)")
    parser.add_argument("--runs-dir", default="runs", help="記録の保存先")
    parser.add_argument("--demo", action="store_true", help="学習後に 1 エピソードを表示する")

    env = parser.add_argument_group("環境")
    env.add_argument("--size", type=int, default=env_defaults.size, help="盤面の一辺のマス数")
    env.add_argument("--max-steps", type=int, default=env_defaults.max_steps)
    env.add_argument("--scramble-moves", type=int, default=env_defaults.scramble_moves,
                     help="開始盤面を作る手数 (省略時は解ける盤面から一様に選ぶ)")

    agent = parser.add_argument_group("エージェント", "ハイパーパラメータは省略すると、選んだ手法ごとの既定値になる")
    agent.add_argument("--method", choices=list(METHODS), default=DEFAULT_METHOD,
                       help=f"学習の手法 (既定: {DEFAULT_METHOD})")
    for name, description in HYPERPARAMETERS.items():
        agent.add_argument(_option(name), type=float, default=None, help=f"{description}。{_default_help(name)}")

    run = parser.add_argument_group("実験")
    run.add_argument("--episodes", type=int, default=settings_defaults.n_episodes, help="学習するエピソード数")
    run.add_argument("--eval-interval", type=int, default=settings_defaults.eval_interval,
                     help="評価の間隔 (エピソード数, 0 なら学習前後のみ)")
    run.add_argument("--eval-episodes", type=int, default=settings_defaults.eval_episodes)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    agent_cls, config_cls = METHODS[args.method]
    supported = {field.name for field in fields(config_cls)}
    hyperparameters = {name: getattr(args, name) for name in HYPERPARAMETERS if getattr(args, name) is not None}
    for name in hyperparameters:
        if name not in supported:
            parser.error(f"{_option(name)} は {args.method} では指定できません (この手法にない設定です)")

    try:
        env = LightsOutEnvironment(
            EnvironmentConfig(
                size=args.size,
                max_steps=args.max_steps,
                scramble_moves=args.scramble_moves,
                seed=args.seed,
            )
        )
        agent = agent_cls(env.n_actions, config_cls(**hyperparameters, seed=args.seed + 1))
        settings = ExperimentSettings(
            name=args.method if args.name is None else args.name,
            n_episodes=args.episodes,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            eval_seed=args.seed + 2,
        )
    except ValueError as error:  # 設定の値の検証エラーは、トレースバックでなく使い方のエラーとして出す
        parser.error(str(error))

    result = ExperimentRunner(FileExperimentRepository(args.runs_dir)).run(env, agent, settings)
    print(f"記録: {args.runs_dir}/{result.run_id}")

    if args.demo:
        _demo(env, agent)
    return 0


def _demo(env: LightsOutEnvironment, agent: Agent) -> None:
    """学習済みの方策で 1 エピソード進め、盤面を表示する。"""
    state = env.reset()
    env.render()
    while not env.is_done():
        action = agent.select_action(state, explore=False)
        row, col = divmod(action, env.size)
        print(f"\n-> マス {action} (行 {row}, 列 {col}) を押す")
        state = env.step(action).state
        env.render()
