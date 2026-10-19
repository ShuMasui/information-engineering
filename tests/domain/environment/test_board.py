import pytest

from lights_out.domain.environment import Board


def lit_cells(board: Board) -> set[int]:
    return {i for i, cell in enumerate(board.cells) if cell}


def test_solved_board_is_all_off():
    board = Board.solved(4)
    assert board.cells == (0,) * 16
    assert board.is_solved()


def test_press_corner_toggles_itself_and_two_neighbors():
    assert lit_cells(Board.solved(4).press(0)) == {0, 1, 4}
    assert lit_cells(Board.solved(4).press(15)) == {15, 14, 11}


def test_press_edge_toggles_four_cells():
    assert lit_cells(Board.solved(4).press(1)) == {0, 1, 2, 5}


def test_press_center_toggles_five_cells():
    assert lit_cells(Board.solved(3).press(4)) == {1, 3, 4, 5, 7}


def test_cells_are_ordered_row_major_from_top_left():
    # 3x3 で 0 行 1 列 (index 1) を押す: 左・右・下が反転する
    assert lit_cells(Board.solved(3).press(1)) == {0, 1, 2, 4}


def test_pressing_twice_restores_the_board():
    board = Board(4, (1, 0, 1, 1) * 4)
    assert board.press(6).press(6) == board


def test_press_returns_new_board_and_keeps_original():
    board = Board.solved(3)
    pressed = board.press(0)
    assert board.is_solved()
    assert not pressed.is_solved()


def test_is_solved_is_false_when_any_cell_is_lit():
    assert not Board(2, (0, 0, 0, 1)).is_solved()


def test_to_list_returns_a_fresh_one_dimensional_list():
    board = Board(2, (1, 0, 0, 1))
    cells = board.to_list()
    assert cells == [1, 0, 0, 1]
    cells[0] = 0
    assert board.cells == (1, 0, 0, 1)


def test_cells_accepts_any_sequence_and_is_hashable():
    board = Board(2, [1, 0, 0, 1])  # type: ignore[arg-type]
    assert board.cells == (1, 0, 0, 1)
    assert hash(board) == hash(Board(2, (1, 0, 0, 1)))


def test_cells_are_normalized_to_int():
    board = Board(2, [True, False, 0.0, 1.0])  # type: ignore[arg-type]
    assert board.cells == (1, 0, 0, 1)
    assert all(type(cell) is int for cell in board.cells)


def test_press_accepts_int_like_indices():
    class NumpyLikeInt:  # np.int64 のように __index__ を持つ値
        def __index__(self):
            return 1

    assert Board.solved(3).press(NumpyLikeInt()) == Board.solved(3).press(1)  # type: ignore[arg-type]


def test_press_rejects_float_index_without_breaking_later_presses():
    with pytest.raises(TypeError):
        Board.solved(4).press(1.0)  # type: ignore[arg-type]
    assert lit_cells(Board.solved(4).press(1)) == {0, 1, 2, 5}


def test_str_shows_one_row_per_line():
    assert str(Board(2, (1, 0, 0, 1))) == "● ○\n○ ●"


@pytest.mark.parametrize("index", [-1, 16])
def test_press_rejects_out_of_range_index(index):
    with pytest.raises(ValueError):
        Board.solved(4).press(index)


def test_rejects_wrong_number_of_cells():
    with pytest.raises(ValueError):
        Board(4, (0,) * 15)


def test_rejects_values_other_than_zero_and_one():
    with pytest.raises(ValueError):
        Board(2, (0, 1, 2, 0))


def test_rejects_non_positive_size():
    with pytest.raises(ValueError):
        Board(0, ())
