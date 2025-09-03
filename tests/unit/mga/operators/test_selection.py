import pytest
import numpy as np
# from numba import njit, int64
# from numba.types import UniTuple
# from numba.experimental import jitclass

from mga.operators import selection as sl

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
    # [True, True, ..., False, False]
    yield np.repeat([True, False], 50) 
    
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

def test_stabilize_sort(niche, objective, rng):
    indices = np.argsort(objective) # order of duplicates not gauranteed 
    sl._stabilize_sort(indices, objective) 
    group_duplicates = indices % 10
    for i in range(10):
        _ind = indices[np.where(group_duplicates == i)]
        assert (_ind == np.sort(_ind)).all(), "stabilize sort does not preserve order of duplicates"



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
            assert (selected == np.arange(9, 101, 10)[-n:]).all(), "check if this an "\
                "issue with stable convention rather than actual stability? "\
                "`stable` only ensures stability but not the specific order"
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

@pytest.mark.parametrize("n", [0, 1, 3, 5, 10])
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
    if n == 0:
        assert selected.size == 0
        return
    if stable:
        if maximize:
            # maximize=True, stable=True
            assert (selected == np.arange(9, 51, 10)[-n:]).all(), "check if this an "\
                "issue with stable convention rather than actual stability? "\
                "`stable` only ensures stability but not the specific order"
        else:
            # maximize=False, stable=True
            assert (selected == np.arange(0, 51, 10)[:n]).all()
    else: 
        if maximize: 
            # maximize=True, stable=False
            assert np.isin(selected, range(9, 51, 10)).all()
        else:
            # maximize=False, stable=False
            assert np.isin(selected, range(0, 51, 10)).all()
    unique, counts = np.unique(selected, return_counts=True)
    assert counts.max() == 1, "_select_best selects duplicates"

