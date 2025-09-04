import pytest 
import numpy as np

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
from mga.problem_definition import OptimizationProblem
import mga.population as pp

# fixtures


# tests
@pytest.mark.parametrize("ndim", [0, 1, 2, 3])
def test_njit_deepcopy(rng, ndim):
    old = rng.random((10,)*ndim)
    new = np.empty_like(old)
    pp.njit_deepcopy(new, old)
    assert (old == new).all(), "copy not performed correctly"
    assert id(old) != id(new), "copy not deep"

# @pytest.mark.parametrize("optimal_obj", [])
# def test_noptimal_threshold_func(optimal_obj, slack, maximize):

# def test_clone
# def test_add_niche_to_array
# def test_find_centroids
# def test_evaluate_noptimality
# def test_evaluate_fitness
# def test_select_parents # shape only
# def test_apply_bounds
# def test_populate_randomly # low priority

## methods 
# def test_init # items, sizes, shapes
# def test_resize
# def test_add_niches
# def test_resize_parent_size
# def test_resize_pop_size
# def test_resize_niche_size
# def test_apply_integrality
# def test_update_optima
# def test_generate_offspring
# def test_evaluate_and_update
# def test_evolve