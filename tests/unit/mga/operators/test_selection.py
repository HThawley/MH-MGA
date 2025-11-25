import pytest
import numpy as np
import pandas as pd
from numba import njit, int64
from numba.types import UniTuple
from numba.experimental import jitclass

from mga.operators import selection as sl


@jitclass([("retval", int64)])
class mock_rng1:
    def __init__(self, n):
        self.retval = n

    def integers(self, lb, ub):
        return self.retval


@jitclass([("retval", int64[:]), ("calls", int64)])
class mock_rngn:
    def __init__(self, n):
        self.calls = -1
        self.retval = n

    def integers(self, lb, ub):
        self.calls += 1
        return self.retval[self.calls]


# fixtures
@pytest.fixture
def niche():
    # [[1], [2], ..., [98], [99]]
    yield np.arange(100).reshape(100, 1).astype(np.float64)


@pytest.fixture
def objective():
    # [1, 2, 3, ..., 8, 9]
    yield np.tile(np.arange(10), 10).astype(np.float64)


@pytest.fixture
def fitness():
    # [1, 2, 3, ..., 18, 19]
    yield np.tile(np.arange(20), 5).astype(np.float64)


@pytest.fixture
def is_noptimal():
    # [False, False, ..., True, True]
    yield np.repeat([False, True], 50)


# tests
@pytest.mark.parametrize("maximize", [True, False])
def test_select_elite(niche, objective, maximize):
    """
    select elite selects best 1 on objective with correct maximize logic

    """
    selected = np.empty((1, 1), np.float64)
    sl.select_elite(selected, niche, objective, maximize)
    if maximize:
        assert selected in range(9, 101, 10)
    else:
        assert selected in range(0, 101, 10)


def test_stabilize_sort(objective, rng):
    index_obj_pairs = np.vstack((np.arange(len(objective)), objective)).T
    for i in range(10):
        rng.shuffle(index_obj_pairs[i * 10 : (i + 1) * 10])

    indices = np.argsort(index_obj_pairs[:, 1])
    sl._stabilize_sort(indices, index_obj_pairs[:, 1])
    index_obj_pairs = index_obj_pairs[indices, :]

    assert (
        index_obj_pairs[:, 1] == np.concatenate([i * np.ones(10, float) for i in range(10)])
    ).all(), "value sort order is bad"
    assert (
        index_obj_pairs[:, 0] == np.concatenate([np.arange(i, 100, 10) for i in range(10)])
    ).all(), "duplicate values not sorted in order of appearance"


@pytest.mark.parametrize("n", [0, 1, 5, 10])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("stable", [True, False])
def test_select_best(niche, objective, n, maximize, stable):
    """
    select best selects best n on objective with correct maximize and stable logic

    triage issues with 'maximize`, 'stable', other by comparing parametrized results
    """
    selected = np.empty((n, niche.shape[1]), niche.dtype)
    sl._select_best(selected, niche, objective, n, maximize, stable)
    selected = selected.flatten()
    if n == 0:
        assert selected.size == 0
        return
    if stable:
        if maximize:
            # maximize=True, stable=True
            assert (selected == np.arange(9, 101, 10)[-n:]).all(), (
                "check if this an "
                "issue with stable convention rather than actual stability? "
                "`stable` only ensures stability but not the specific order"
            )
        else:
            # maximize=False, stable=True
            assert (selected == np.arange(0, 101, 10)[:n]).all()
    else:
        if maximize:
            # maximize=True, stable=False
            assert np.isin(selected, range(9, 101, 10)).all()
        else:
            # maximize=False, stable=False
            assert np.isin(selected, range(0, 101, 10)).all()
    unique, counts = np.unique(selected, return_counts=True)
    assert counts.max() == 1, "_select_best selects duplicates"


@pytest.mark.parametrize("n", [0, 1, 5, 49, 50, 51])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("stable", [True, False])
def test_select_best_with_fallback(niche, fitness, is_noptimal, objective, n, maximize, stable):
    """
    select best selects best n on objective with correct maximize and stable logic

    triage issues with 'maximize`, 'stable', other by comparing parametrized results
    """
    selected = np.empty((n, niche.shape[1]), niche.dtype)
    sl._select_best_with_fallback(selected, niche, fitness, is_noptimal, objective, n, maximize, stable)
    selected = selected.flatten()
    selected_idx = selected.astype(int)
    if n == 0:
        assert selected.size == 0
        return

    _, counts = np.unique(selected_idx, return_counts=True)
    assert counts.max() == 1, "_select_best_with_fallback selects duplicates"

    points = pd.DataFrame(
        {
            "indices": np.arange(len(niche)),
            "fitness": fitness,
            "is_noptimal": is_noptimal,
            "objective": objective,
        }
    )

    if n >= is_noptimal.sum():  # should select on objective
        selected_objectives = objective[selected_idx]
        vc = pd.DataFrame(points.loc[:, "objective"].value_counts()).sort_index(ascending=(not maximize))
        worst_objective = vc.iloc[np.where(vc["count"].cumsum() >= n)[0][0]].name
        points = points.sort_values(["objective", "indices"], ascending=True)

        if maximize:
            assert selected_objectives.min() >= worst_objective, "did not select best objectives"
            selectable = points.loc[points["objective"] >= worst_objective, "indices"]
            stable_selectable = selectable[-n:]
        else:
            assert selected_objectives.max() <= worst_objective, "did not select best objectives"
            selectable = points.loc[points["objective"] <= worst_objective, "indices"]
            stable_selectable = selectable[:n]

    else:  # select on fitness  # fitness is always maximized so 'maximize' is ignored

        selected_fitnesses = fitness[selected_idx]
        vc = pd.DataFrame(points.loc[points["is_noptimal"], "fitness"].value_counts()).sort_index(ascending=False)
        worst_fitness = vc.iloc[np.where(vc["count"].cumsum() >= n)[0][0]].name

        assert selected_fitnesses.min() >= worst_fitness, "did not select best noptimal fitnesses"
        assert is_noptimal[selected_idx].all(), "selected non-noptimal points without fallback"

        points = points.sort_values(["fitness", "indices"], ascending=True)
        selectable = points.loc[(points["is_noptimal"]) & (points["fitness"] >= worst_fitness), "indices"]
        stable_selectable = selectable[-n:]

    assert np.isin(selected_idx, selectable).all(), "selected bad indices"
    if stable:
        assert (selected_idx == stable_selectable).all(), "did not select indices in correct order (stable=True)"


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("indices", [np.arange(10), np.array([0, 5, 9]), np.array([0, 10, 20])])
def test_do_tournament(objective, maximize, indices):
    _selected_idx = sl._do_tournament(objective, maximize, indices)
    # picks index of the best
    if maximize:
        assert _selected_idx == indices[np.where(objective[indices] == objective[indices].max())[0]][0]
    else:
        assert _selected_idx == indices[np.where(objective[indices] == objective[indices].min())[0]][0]


@pytest.mark.parametrize("maximize", [True, False])
def test_select_elite_with_fallback(niche, fitness, is_noptimal, objective, maximize):
    selected = np.empty((1, niche.shape[1]), np.float64)
    sl.select_elite_with_fallback(selected, niche, fitness, is_noptimal, objective, maximize)
    idx = int(selected[0, 0])
    if is_noptimal.any():
        # should select by best noptimal fitness
        assert is_noptimal[idx]
        assert fitness[idx] == fitness[is_noptimal].max()
    else:
        # fallback to objective
        if maximize:
            assert objective[idx] == objective.max()
        else:
            assert objective[idx] == objective.min()


@pytest.mark.parametrize("n", [0, 1, 5])
@pytest.mark.parametrize("maximize", [True, False])
def test_select_tournament(niche, objective, rng, n, maximize):
    selected = np.empty((n, niche.shape[1]), np.float64)
    sl._select_tournament(selected, niche, objective, n, 5, rng, maximize)
    assert selected.shape[0] == n
    for row in selected:
        assert row[0] in niche.flatten()


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("maximize", [True, False])
def test_select_tournament_mechanism(niche, objective, n, maximize, rng):
    tourn_size = 4
    draws = rng.integers(0, niche.shape[0], tourn_size * n)
    rng = mock_rngn(draws)
    selected = np.empty((n, 1), np.float64)
    sl._select_tournament(selected, niche, objective, n, tourn_size, rng, maximize)
    expected = []
    for round_i in range(n):
        round_draws = draws[round_i * tourn_size : (round_i + 1) * tourn_size]
        if maximize:
            expected_idx = round_draws[np.argmax(objective[round_draws])]
        else:
            expected_idx = round_draws[np.argmin(objective[round_draws])]
        expected.append(niche[expected_idx, 0])
    assert np.allclose(selected.flatten(), expected)


@pytest.mark.parametrize("n", [0, 1, 5])
@pytest.mark.parametrize("maximize", [True, False])
def test_select_tournament_with_fallback(niche, fitness, is_noptimal, objective, rng, n, maximize):
    selected = np.empty((n, niche.shape[1]), np.float64)
    sl._select_tournament_with_fallback(selected, niche, fitness, is_noptimal, objective, n, 5, rng, maximize)
    assert selected.shape[0] == n
    if n > 0:
        idx = selected.astype(int).flatten()
        assert np.isin(idx, np.arange(len(niche))).all()


@pytest.mark.parametrize(
    "draws, expect_fallback",
    [
        (np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14], dtype=np.int64), True),
        (np.array([50, 51, 52, 53, 54, 60, 61, 62, 63, 64], dtype=np.int64), False),
    ],
)
@pytest.mark.parametrize("maximize", [True, False])
def test_select_tournament_with_fallback_mechanism(
    niche, fitness, is_noptimal, objective, draws, expect_fallback, maximize
):
    rng = mock_rngn(draws)
    n = 2
    tourn_size = 5
    selected = np.empty((n, niche.shape[1]), np.float64)
    sl._select_tournament_with_fallback(selected, niche, fitness, is_noptimal, objective, n, tourn_size, rng, maximize)

    for round_i in range(n):
        round_draws = draws[round_i * tourn_size : (round_i + 1) * tourn_size]
        if expect_fallback:
            if maximize:
                expected_idx = round_draws[np.argmax(objective[round_draws])]
            else:
                expected_idx = round_draws[np.argmin(objective[round_draws])]
        else:
            expected_idx = round_draws[np.argmax(fitness[round_draws])]
        assert int(selected[round_i, 0]) == int(niche[expected_idx, 0])


@pytest.mark.parametrize("maximize", [True, False])
def test_selection(niche, objective, rng, maximize):
    elite_count, tourn_count = 3, 5
    selected = np.empty((elite_count + tourn_count, niche.shape[1]), np.float64)
    sl.selection(selected, niche, objective, maximize, elite_count, tourn_count, 3, rng, True)
    assert selected.shape[0] == elite_count + tourn_count
    elite = selected[:elite_count]
    tourn = selected[elite_count:]
    # elites must be best by objective
    if maximize:
        assert objective[elite.flatten().astype(int)].min() >= np.partition(objective, -elite_count)[-elite_count]
    else:
        assert objective[elite.flatten().astype(int)].max() <= np.partition(objective, elite_count)[:elite_count].max()
    assert tourn.shape[0] == tourn_count


@pytest.mark.parametrize("maximize", [True, False])
def test_selection_with_fallback(niche, fitness, is_noptimal, objective, maximize):
    elite_count, tourn_count = 2, 4
    selected = np.empty((elite_count + tourn_count, 1), np.float64)
    rng = sl.np.random.default_rng(3)
    sl.selection_with_fallback(
        selected, niche, fitness, is_noptimal, objective, maximize, elite_count, tourn_count, 3, rng, True
    )
    assert selected.shape[0] == elite_count + tourn_count
    idx = selected.astype(int).flatten()
    assert np.isin(idx, np.arange(len(niche))).all()
