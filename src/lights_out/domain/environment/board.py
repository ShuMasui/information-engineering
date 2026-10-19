"""盤面 (値オブジェクト)。"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import lru_cache

OFF = 0
LIT = 1

_SYMBOLS = {OFF: "○", LIT: "●"}
_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@lru_cache(maxsize=None)
def _toggled_cells(size: int, index: int) -> tuple[int, ...]:
    """index のマスを押したときに反転するマス (自身と上下左右)。"""
    row, col = divmod(index, size)
    cells = [index]
    for d_row, d_col in _NEIGHBOR_OFFSETS:
        r, c = row + d_row, col + d_col
        if 0 <= r < size and 0 <= c < size:
            cells.append(r * size + c)
    return tuple(cells)


@dataclass(frozen=True)
class Board:
    """size × size の盤面。

    マスは一次元の ``cells`` に左上から右下へ (行優先で) 並べる。
    ``cells[row * size + col]`` が 1 なら点灯、0 なら消灯。
    不変なので、``press`` は新しい ``Board`` を返す。
    """

    size: int
    cells: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"size は 1 以上にしてください: {self.size}")
        cells = tuple(self.cells)
        if len(cells) != self.size**2:
            raise ValueError(f"cells の長さは size**2 = {self.size**2} 個にしてください: {len(cells)}")
        if any(cell not in (OFF, LIT) for cell in cells):
            raise ValueError(f"cells の値は 0 か 1 にしてください: {cells}")
        # True や 1.0 が渡されても、盤面は常に int の 0/1 で持つ
        object.__setattr__(self, "cells", tuple(int(cell) for cell in cells))

    @classmethod
    def solved(cls, size: int) -> Board:
        """全マス消灯 (クリア状態) の盤面。"""
        return cls(size, (OFF,) * size**2)

    @property
    def n_cells(self) -> int:
        return self.size**2

    def press(self, index: int) -> Board:
        """index のマスを押す。自身と上下左右のマスが反転する。"""
        index = operator.index(index)  # float などは lru_cache に入る前に TypeError で弾く
        if not 0 <= index < self.n_cells:
            raise ValueError(f"index は 0 以上 {self.n_cells} 未満にしてください: {index}")
        cells = list(self.cells)
        for i in _toggled_cells(self.size, index):
            cells[i] ^= 1
        return Board(self.size, tuple(cells))

    def is_solved(self) -> bool:
        """全マスが消灯しているか。"""
        return not any(self.cells)

    def to_list(self) -> list[int]:
        """盤面を一次元の配列 (左上 → 右下) として返す。呼び出しごとに新しい list。"""
        return list(self.cells)

    def __str__(self) -> str:
        return "\n".join(
            " ".join(_SYMBOLS[cell] for cell in self.cells[row * self.size : (row + 1) * self.size])
            for row in range(self.size)
        )
