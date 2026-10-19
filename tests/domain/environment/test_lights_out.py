import pytest

from lights_out.domain.environment import EnvironmentConfig, LightsOutEnvironment


def make_env(**kwargs) -> LightsOutEnvironment:
    return LightsOutEnvironment(EnvironmentConfig(**kwargs))


def is_solvable(size: int, state: list[int]) -> bool:
    """GF(2) のガウスの消去法で、state が全消灯にできるか (= 押す組み合わせが存在するか) を調べる。"""
    n = size * size
    # 列 j = マス j を押したときの反転パターン (ビットマスク)
    columns = []
    for j in range(n):
        mask = 0
        row, col = divmod(j, size)
        for r, c in ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= r < size and 0 <= c < size:
                mask |= 1 << (r * size + c)
        columns.append(mask)
    target = sum(bit << i for i, bit in enumerate(state))
    basis: dict[int, int] = {}  # 最上位ビット -> 基底ベクトル
    for vec in columns:
        while vec:
            top = vec.bit_length() - 1
            if top not in basis:
                basis[top] = vec
                break
            vec ^= basis[top]
    while target:
        top = target.bit_length() - 1
        if top not in basis:
            return False
        target ^= basis[top]
    return True


def test_default_environment_is_4x4():
    env = LightsOutEnvironment()
    assert env.size == 4
    assert env.n_actions == 16
    assert len(env.get_state()) == 16


@pytest.mark.parametrize("size", [1, 2, 3, 5, 6])
def test_board_size_is_configurable(size):
    env = make_env(size=size, seed=0)
    assert env.n_actions == size**2
    assert len(env.get_state()) == size**2


def test_state_is_a_one_dimensional_list_of_zero_and_one():
    state = make_env(seed=0).get_state()
    assert isinstance(state, list)
    assert set(state) <= {0, 1}


def test_state_is_ordered_from_top_left_to_bottom_right():
    env = make_env(size=3, seed=0)
    env.reset([1, 0, 0, 0, 0, 0, 0, 0, 0])
    # 左上 (0) だけ点灯 → 0 を押すと 0,1,3 が反転して 1,3 が点灯
    assert env.step(0).state == [0, 1, 0, 1, 0, 0, 0, 0, 0]


def test_get_state_returns_a_copy():
    env = make_env(seed=0)
    state = env.get_state()
    state[0] ^= 1
    assert env.get_state() != state


def test_new_environment_is_at_start_and_not_done():
    env = make_env(seed=0)
    assert env.is_start()
    assert not env.is_solved()
    assert not env.is_truncated()
    assert not env.is_done()
    assert env.steps == 0


def test_reset_returns_the_new_state_and_returns_to_start():
    env = make_env(seed=0)
    env.step(0)
    assert not env.is_start()
    state = env.reset()
    assert state == env.get_state()
    assert env.is_start()
    assert env.steps == 0


def test_step_presses_the_cell_and_counts_the_step():
    env = make_env(size=3, seed=0)
    env.reset([1, 1, 0, 0, 0, 0, 0, 0, 0])
    result = env.step(2)
    assert result.state == [1, 0, 1, 0, 0, 1, 0, 0, 0]
    assert result.state == env.get_state()
    assert env.steps == 1
    assert not env.is_start()


def test_non_clearing_step_gives_step_reward():
    env = make_env(step_reward=-2.0, goal_reward=50.0, seed=0)
    result = env.step(0)
    assert result.reward == -2.0
    assert not result.terminated


def test_clearing_step_terminates_with_goal_reward():
    env = make_env(size=3, goal_reward=50.0, seed=0)
    env.reset([1, 1, 0, 1, 0, 0, 0, 0, 0])  # 0 を押すと全消灯
    result = env.step(0)
    assert result.terminated
    assert not result.truncated
    assert result.done
    assert result.reward == 50.0
    assert env.is_solved()
    assert env.is_done()


def test_step_after_episode_end_raises():
    env = make_env(size=3, seed=0)
    env.reset([1, 1, 0, 1, 0, 0, 0, 0, 0])
    env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_episode_is_truncated_at_max_steps():
    env = make_env(size=3, max_steps=2, seed=0)
    env.reset([1, 1, 0, 1, 0, 0, 0, 0, 0])
    first = env.step(8)
    assert not first.done
    second = env.step(8)  # 同じマスを 2 回押して元に戻る
    assert second.truncated
    assert not second.terminated
    assert second.reward == -1.0  # 打ち切りの手はクリア報酬ではない
    assert second.done
    assert env.is_truncated()
    assert env.is_done()
    assert not env.is_solved()


def test_clearing_on_the_last_step_is_terminated_not_truncated():
    env = make_env(size=3, max_steps=1, seed=0)
    env.reset([1, 1, 0, 1, 0, 0, 0, 0, 0])
    result = env.step(0)
    assert result.terminated
    assert not result.truncated
    assert not env.is_truncated()


def test_reset_after_truncation_starts_a_new_episode():
    env = make_env(size=3, max_steps=1, seed=0)
    env.reset([1, 1, 0, 1, 0, 0, 0, 0, 0])
    env.step(8)
    assert env.is_done()
    env.reset()
    assert not env.is_done()
    assert env.is_start()


def test_step_rejects_out_of_range_action():
    env = make_env(seed=0)
    with pytest.raises(ValueError):
        env.step(16)
    assert env.steps == 0


def test_reset_with_initial_state_uses_it():
    env = make_env(size=2, seed=0)
    assert env.reset([1, 0, 0, 1]) == [1, 0, 0, 1]
    assert env.get_state() == [1, 0, 0, 1]


def test_reset_rejects_already_solved_initial_state():
    with pytest.raises(ValueError):
        make_env(size=2, seed=0).reset([0, 0, 0, 0])


def test_reset_rejects_initial_state_of_wrong_length():
    with pytest.raises(ValueError):
        make_env(size=2, seed=0).reset([1, 0, 0])


def test_same_seed_gives_same_start_states():
    a, b = make_env(seed=42), make_env(seed=42)
    assert [a.reset() for _ in range(20)] == [b.reset() for _ in range(20)]


def test_different_seeds_give_different_start_states():
    a, b = make_env(seed=1), make_env(seed=2)
    assert [a.reset() for _ in range(20)] != [b.reset() for _ in range(20)]


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_start_states_are_always_solvable_and_not_solved(size):
    env = make_env(size=size, seed=0)
    for _ in range(200):
        state = env.reset()
        assert any(state)
        assert is_solvable(size, state)


def test_solvability_helper_detects_unsolvable_boards():
    # 4x4 の解ける盤面は 2**12 通りで、全 2**16 通りの一部だけ
    assert is_solvable(4, [0] * 16)
    assert not is_solvable(4, [1] + [0] * 15)


def test_start_states_cover_many_solvable_boards():
    env = make_env(size=4, seed=0)
    assert len({tuple(env.reset()) for _ in range(2000)}) > 1000


def test_scramble_moves_one_starts_one_press_away_from_cleared():
    env = make_env(size=4, scramble_moves=1, seed=0)
    for _ in range(30):
        start = env.reset()
        clearing = [a for a in range(16) if _clears(start, a)]
        assert len(clearing) == 1


def _clears(start: list[int], action: int) -> bool:
    env = make_env(seed=0)
    env.reset(start)
    return env.step(action).terminated


def test_scramble_moves_bounds_the_solution_length():
    env = make_env(size=4, scramble_moves=3, seed=0)
    for _ in range(50):
        assert is_solvable(4, env.reset())


def test_render_prints_status_and_board_to_stdout(capsys):
    env = make_env(size=2, max_steps=10, seed=0)
    env.reset([1, 0, 0, 1])
    env.step(0)
    env.render()
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "step 1/10  solved: False"
    assert out.splitlines()[1:] == str(env._board).splitlines()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 0},
        {"max_steps": 0},
        {"scramble_moves": 0},
        {"scramble_moves": 17},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        EnvironmentConfig(**kwargs)
