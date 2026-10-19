"""コマンドライン: Q学習エージェントでライツアウトを学習する。

    python -m lights_out --episodes 10000 --seed 0
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from lights_out.application import ExperimentRunner, ExperimentSettings
from lights_out.domain.agent import QLearningAgent, QLearningConfig
from lights_out.domain.environment import EnvironmentConfig, LightsOutEnvironment
from lights_out.infrastructure import FileExperimentRepository


def build_parser() -> argparse.ArgumentParser:
    env_defaults = EnvironmentConfig()
    agent_defaults = QLearningConfig()
    settings_defaults = ExperimentSettings()

    parser = argparse.ArgumentParser(
        prog="lights-out", description="ライツアウトをQ学習で学習し、結果を runs/ に記録する"
    )
    parser.add_argument("--seed", type=int, default=0, help="基準シード (環境=seed, エージェント=seed+1, 評価=seed+2)")
    parser.add_argument("--name", default=settings_defaults.name, help="run の名前")
    parser.add_argument("--runs-dir", default="runs", help="記録の保存先")
    parser.add_argument("--demo", action="store_true", help="学習後に 1 エピソードを表示する")

    env = parser.add_argument_group("環境")
    env.add_argument("--size", type=int, default=env_defaults.size, help="盤面の一辺のマス数")
    env.add_argument("--max-steps", type=int, default=env_defaults.max_steps)
    env.add_argument("--scramble-moves", type=int, default=env_defaults.scramble_moves,
                     help="開始盤面を作る手数 (省略時は解ける盤面から一様に選ぶ)")

    agent = parser.add_argument_group("Q学習")
    agent.add_argument("--learning-rate", type=float, default=agent_defaults.learning_rate)
    agent.add_argument("--discount-factor", type=float, default=agent_defaults.discount_factor)
    agent.add_argument("--epsilon-start", type=float, default=agent_defaults.epsilon_start)
    agent.add_argument("--epsilon-min", type=float, default=agent_defaults.epsilon_min)
    agent.add_argument("--epsilon-decay", type=float, default=agent_defaults.epsilon_decay)

    run = parser.add_argument_group("実験")
    run.add_argument("--episodes", type=int, default=settings_defaults.n_episodes, help="学習するエピソード数")
    run.add_argument("--eval-interval", type=int, default=settings_defaults.eval_interval,
                     help="評価の間隔 (エピソード数, 0 なら学習前後のみ)")
    run.add_argument("--eval-episodes", type=int, default=settings_defaults.eval_episodes)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    env = LightsOutEnvironment(
        EnvironmentConfig(
            size=args.size,
            max_steps=args.max_steps,
            scramble_moves=args.scramble_moves,
            seed=args.seed,
        )
    )
    agent = QLearningAgent(
        env.n_actions,
        QLearningConfig(
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed + 1,
        ),
    )
    settings = ExperimentSettings(
        name=args.name,
        n_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_seed=args.seed + 2,
    )

    result = ExperimentRunner(FileExperimentRepository(args.runs_dir)).run(env, agent, settings)
    print(f"記録: {args.runs_dir}/{result.run_id}")

    if args.demo:
        _demo(env, agent)
    return 0


def _demo(env: LightsOutEnvironment, agent: QLearningAgent) -> None:
    """学習済みの方策で 1 エピソード進め、盤面を表示する。"""
    state = env.reset()
    env.render()
    while not env.is_done():
        action = agent.select_action(state, explore=False)
        row, col = divmod(action, env.size)
        print(f"\n-> マス {action} (行 {row}, 列 {col}) を押す")
        state = env.step(action).state
        env.render()
