"""ライツアウト環境 (エンティティ)。"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from lights_out.domain.environment.board import Board

# scramble の結果がクリア状態になったときに引き直す上限回数 (無限ループ防止)
_MAX_RESET_ATTEMPTS = 1000


@dataclass(frozen=True)
class EnvironmentConfig:
    """環境の設定。実験の記録にそのまま残せるよう、値だけを持つ。

    Attributes:
        size: 盤面の一辺のマス数 (size × size)。
        max_steps: 1 エピソードの最大手数。超えると打ち切り (truncated)。
        scramble_moves: 開始盤面を作るために、クリア状態から押すマスの数。
            None なら、各マスを 1/2 の確率で押す (解ける盤面を一様に選ぶ)。
            どちらの場合も開始盤面は必ず解ける。
        step_reward: 1 手ごとの報酬 (クリアに至らなかった手)。
        goal_reward: クリアした手の報酬。
        seed: 開始盤面の乱数シード。None なら非決定的。
    """

    size: int = 4
    max_steps: int = 100
    scramble_moves: int | None = None
    step_reward: float = -1.0
    goal_reward: float = 100.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"size は 1 以上にしてください: {self.size}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps は 1 以上にしてください: {self.max_steps}")
        if self.scramble_moves is not None and not 1 <= self.scramble_moves <= self.size**2:
            raise ValueError(
                f"scramble_moves は 1 以上 {self.size**2} 以下にしてください: {self.scramble_moves}"
            )


@dataclass(frozen=True)
class StepResult:
    """``step`` の結果。

    ``terminated`` は本来の終了 (クリア)、``truncated`` は最大手数による打ち切り。
    学習での扱いは手法による (Q 学習・SARSA は打ち切りの先にも価値があるとして更新し、
    モンテカルロ法は打ち切りもエピソードの終わりとして扱う。詳しくは README)。
    """

    state: list[int]
    reward: float
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class LightsOutEnvironment:
    """ライツアウトの環境。盤面と手数の状態をこのクラスが保持する。

    マスの番号 (= 行動) は左上を 0、右下を size**2 - 1 とする行優先の一次元で表す。
    生成直後に ``reset`` 済みなので、すぐに ``step`` できる。
    """

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self._config = config or EnvironmentConfig()
        self._rng = random.Random(self._config.seed)
        self._board: Board
        self._steps: int
        self.reset()

    # ----- 属性 -----

    @property
    def config(self) -> EnvironmentConfig:
        return self._config

    @property
    def size(self) -> int:
        return self._config.size

    @property
    def n_actions(self) -> int:
        """行動数 (= マス数)。"""
        return self._config.size**2

    @property
    def steps(self) -> int:
        """現在のエピソードで押した回数。"""
        return self._steps

    # ----- 操作 -----

    def reset(self, initial_state: Sequence[int] | None = None) -> list[int]:
        """新しいエピソードを開始し、開始盤面を返す。

        initial_state を渡すとその盤面から開始する (評価や再現用)。
        省略時は、クリア状態からランダムに押して作る (必ず解ける・クリア済みにはならない)。
        """
        if initial_state is None:
            board = self._random_start()
        else:
            board = Board(self.size, tuple(initial_state))
            if board.is_solved():
                raise ValueError("開始盤面がすでにクリア状態です")
        self._board = board
        self._steps = 0
        return self.get_state()

    def step(self, action: int) -> StepResult:
        """action のマスを押して 1 手進める。終了したエピソードでは呼べない。"""
        if self.is_done():
            raise RuntimeError("エピソードは終了しています。reset() を呼んでください")
        self._board = self._board.press(action)
        self._steps += 1
        terminated = self.is_solved()
        reward = self._config.goal_reward if terminated else self._config.step_reward
        return StepResult(
            state=self.get_state(),
            reward=reward,
            terminated=terminated,
            truncated=self.is_truncated(),
        )

    # ----- 状態 -----

    def get_state(self) -> list[int]:
        """盤面を一次元の配列 (左上 → 右下) で返す。1 が点灯、0 が消灯。"""
        return self._board.to_list()

    # ----- 判定 -----

    def is_start(self) -> bool:
        """エピソードの開始状態か (reset 直後でまだ 1 手も押していない)。"""
        return self._steps == 0

    def is_solved(self) -> bool:
        """クリアしたか (全マス消灯)。"""
        return self._board.is_solved()

    def is_truncated(self) -> bool:
        """クリアできないまま最大手数に達したか。"""
        return not self.is_solved() and self._steps >= self._config.max_steps

    def is_done(self) -> bool:
        """エピソードの終了判定 (クリア or 打ち切り)。"""
        return self.is_solved() or self.is_truncated()

    # ----- 出力 -----

    def render(self) -> None:
        """現在の環境を標準出力に表示する。"""
        print(f"step {self._steps}/{self._config.max_steps}  solved: {self.is_solved()}")
        print(self._board)

    # ----- 内部 -----

    def _random_start(self) -> Board:
        for _ in range(_MAX_RESET_ATTEMPTS):
            board = self._scramble()
            if not board.is_solved():
                return board
        raise RuntimeError(f"{_MAX_RESET_ATTEMPTS} 回試してもクリア済みでない開始盤面を作れませんでした")

    def _scramble(self) -> Board:
        n_cells = self.n_actions
        if self._config.scramble_moves is None:
            presses = [i for i in range(n_cells) if self._rng.random() < 0.5]
        else:
            presses = self._rng.sample(range(n_cells), self._config.scramble_moves)
        board = Board.solved(self.size)
        for index in presses:
            board = board.press(index)
        return board
