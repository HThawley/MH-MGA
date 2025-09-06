import pytest 
import numpy as np

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
# FLOAT == np.float64
# INT == np.int64
from mga.problem_definition import OptimizationProblem
import mga.population as pp

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
# def test_resize_parent_size
# def test_resize_niche_size
# def test_generate_offspring
# def test_evaluate_and_update
# def test_evolve

class MockOptimizationProblem:
    def __init__(self, ndim=3, maximize=True):
        self.ndim = ndim
        self.lower_bounds = np.zeros(ndim, dtype=FLOAT)
        self.upper_bounds = np.ones(ndim, dtype=FLOAT)
        self.integrality = np.zeros(ndim, dtype=bool)
        self.boolean_mask = np.zeros(ndim, dtype=bool)
        self.known_optimum_point = np.full(ndim, 0.5, dtype=FLOAT)
        self.known_optimum_value = 1.0 if maximize else 0.0
        self.maximize = maximize

    def evaluate(self, points):
        # Simple evaluation: sum of squares, negated for maximization
        objectives = 2-np.sum(points**2, axis=1) if self.maximize else np.sum(points**2, axis=1)
        violations = np.zeros(points.shape[0])
        return objectives, violations

# Fixtures
@pytest.fixture
def mock_problem():
    """Provides a mock OptimizationProblem instance."""
    yield MockOptimizationProblem(ndim=3)

@pytest.fixture
def population_instance(mock_problem, rng):
    """Provides a Population instance for testing."""
    yield pp.Population(
        problem=mock_problem, 
        num_niches=2, 
        pop_size=5, 
        rng=rng, 
        stable_sort=False,
        )

# Tests for @njit helper functions

@pytest.mark.parametrize("ndim", [0, 1, 2, 3])
def test_njit_deepcopy(rng, ndim):
    old = rng.random((10,)*ndim)
    new = np.empty_like(old)
    pp.njit_deepcopy(new, old)
    assert np.array_equal(old, new), "copy not performed correctly"
    if old.size > 0:
        assert id(old) != id(new), "copy not deep"

@pytest.mark.parametrize("optimal_obj, slack, maximize, expected", [
    (100.0, 1.1, False, 110.0),
    (100.0, 1.1, True, 90.0),
    (50.0, 1.5, False, 75.0),
    (50.0, 1.5, True, 25.0),
])
def test_noptimal_threshold(optimal_obj, slack, maximize, expected):
    """Tests the near-optimal threshold calculation."""
    result = pp._noptimal_threshold(optimal_obj, slack, maximize)
    assert np.isclose(result, expected)

def test_clone(rng):
    """Tests the cloning of individuals into a larger population."""
    start_pop = rng.random((2, 3, 4)) # niches, pop_size, ndim
    target = np.empty((2, 7, 4))
    pp._clone(target, start_pop)
    # Check if the first 3 individuals are copied correctly
    assert np.array_equal(target[:, :3, :], start_pop)
    # Check if the cloning wraps around correctly
    assert np.array_equal(target[:, 3, :], start_pop[:, 0, :])
    assert np.array_equal(target[:, 6, :], start_pop[:, 0, :])

@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("new_niches", [-1, 0, 1])
def test_add_niche_to_array(rng, ndim, new_niches):
    """Tests resizing an array to add new niches."""
    shape = [5] * ndim
    old_niches = shape[0]
    total_niches = old_niches + new_niches
    old_array = rng.random(tuple(shape))
    if new_niches < 0:
        with pytest.raises(Exception):
            new_array = pp._add_niche_to_array(old_array, total_niches)
        return
    new_array = pp._add_niche_to_array(old_array, total_niches)
    assert new_array.shape[0] == total_niches
    assert new_array.shape[1:] == old_array.shape[1:]
    # Check if original data is preserved
    assert np.array_equal(old_array, new_array[:old_niches])

def test_find_centroids():
    """Tests the centroid calculation."""
    points = np.array([
        [[1, 2, 3], [3, 4, 5]], # Niche 1
        [[5, 6, 7], [7, 8, 9]], # Niche 2
    ], dtype=FLOAT)
    centroids = np.empty((2, 3), dtype=FLOAT)
    pp._find_centroids(centroids, points)
    expected_centroids = np.array([[2, 3, 4], [6, 7, 8]])
    assert np.allclose(centroids, expected_centroids)

@pytest.mark.parametrize("maximize", [True, False])
def test_evaluate_noptimality(maximize):
    """Tests the evaluation of near-optimality."""
    objectives = np.array([[10, 20, 30], [5, 15, 25]])
    threshold = 18
    is_noptimal = np.empty_like(objectives, dtype=np.bool_)
    pp._evaluate_noptimality(is_noptimal, objectives, threshold, maximize)
    if maximize:
        expected = np.array([[False, True, True], [False, False, True]])
    else:
        expected = np.array([[True, False, False], [True, True, False]])
    assert np.array_equal(is_noptimal, expected)

def test_apply_bounds(rng):
    """Tests if points are correctly clipped to bounds."""
    points = rng.random((2, 5, 3)) * 2 - 0.5 # Values between -0.5 and 1.5
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    pp._apply_bounds(points, lb, ub)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)


# Tests for Population class methods

def test_init(population_instance, mock_problem):
    """Tests the initialization of the Population class."""
    pop = population_instance
    assert pop.num_niches == 2
    assert pop.pop_size == 5
    assert pop.points.shape == (2, 5, mock_problem.ndim)
    assert pop.objective_values.shape == (2, 5)
    assert pop.centroids.shape == (2, mock_problem.ndim)
    assert np.all(pop.violations == 0)

def test_populate(population_instance, mock_problem):
    """Tests random population generation."""
    pop = population_instance
    pop.populate()
    assert np.all(pop.points >= mock_problem.lower_bounds)
    assert np.all(pop.points <= mock_problem.upper_bounds)

def test_add_niches(population_instance):
    """Tests adding new niches to the population."""
    pop = population_instance
    initial_niches = pop.num_niches
    pop.add_niches(3)
    assert pop.num_niches == initial_niches + 3
    assert pop.points.shape[0] == initial_niches + 3
    with pytest.raises(ValueError):
        pop.add_niches(-1)

def test_resize_pop_size(population_instance):
    """Tests resizing the population size of each niche."""
    pop = population_instance
    pop.populate()
    pop.evaluate_and_update(1.1, 1.0) # Need fitness values for resizing down
    
    # Increase size
    pop.resize(pop_size=10)
    assert pop.pop_size == 10
    assert pop.points.shape == (2, 10, 3)

    # Decrease size
    pop._resize_parent_size(5)
    pop.elite_count, pop.tourn_count, pop.tourn_size=5, 0, 0
    pop.resize(pop_size=4)
    assert pop.pop_size == 4
    assert pop.points.shape == (2, 4, 3)

def test_apply_integrality(population_instance):
    """Tests the enforcement of integrality constraints."""
    pop = population_instance
    pop.problem.integrality[1] = True  # Make the second dimension integer
    pop.points[:] = 0.4
    pop._apply_integrality()
    assert np.all(pop.points[:, :, 0] == 0.4)
    assert np.all(pop.points[:, :, 1] == 0.0) # Should be rounded
    assert np.all(pop.points[:, :, 2] == 0.4)

@pytest.mark.parametrize("niche_idx", [0, 1])
def test_update_optima_minimize(population_instance, niche_idx):
    """Tests if the global optimum is updated correctly for minimization."""
    pop = population_instance
    pop.problem.maximize=False
    pop.populate()
    
    # Manually set a point to be better than the known optimum
    pop.points[niche_idx, 3, :] = np.array([0.01, 0.01, 0.01])
    initial_optimum = pop.current_optima_obj[0]
    
    pop.evaluate_and_update(noptimal_slack=1.1, violation_factor=1.0)
    
    # New optimum should be sum of squares of [0.01, 0.01, 0.01] = 0.0003
    assert pop.current_optima_obj[0] <= initial_optimum
    assert np.isclose(pop.current_optima_obj[0], 0.0003)
    assert np.allclose(pop.current_optima[0], [0.01, 0.01, 0.01])

@pytest.mark.parametrize("niche_idx", [0, 1])
def test_update_optima_maximize(population_instance, niche_idx):
    """Tests if the global optimum is updated correctly for maximization."""
    pop = population_instance
    pop.problem.maximize=True
    pop.populate()

    # Manually set a point with a high objective value (close to zero)
    pop.points[niche_idx, 3, :] = np.array([0.01, 0.01, 0.01])
    initial_optimum = pop.current_optima_obj[0] 
    
    pop.evaluate_and_update(noptimal_slack=1.1, violation_factor=1.0)
    
    # New optimum should be offset -sum of squares of [0.01, 0.01, 0.01] = 2-0.0003
    assert pop.current_optima_obj[0] >= initial_optimum
    assert np.isclose(pop.current_optima_obj[0], 2-0.0003)
    assert np.allclose(pop.current_optima[0], [0.01, 0.01, 0.01])

def test_evolve(population_instance):
    """Tests a single evolution step."""
    pop = population_instance
    pop.populate()
    pop.evaluate_and_update(1.1, 1.0)
    
    initial_points = pop.points.copy()
    pop.resize(parent_size=3)
    pop.evolve(
        elite_count=1,
        tourn_count=2,
        tourn_size=2,
        mutation_prob=1.0,
        mutation_sigma=0.1,
        crossover_prob=0.8,
        niche_elitism="selfish",
    )
    
    # The points should have changed after evolution
    assert not np.array_equal(initial_points, pop.points)
    #TODO assertions about internal shapes
    # The best individual (optima) should be preserved
    assert np.allclose(pop.points[0, 0, :], pop.current_optima[0])
