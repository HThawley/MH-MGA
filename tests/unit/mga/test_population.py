import pytest 
import numpy as np

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
import mga.population as pp

class MockOptimizationProblem:
    def __init__(self, ndim=3, maximize=True):
        self.ndim = ndim
        self.lower_bounds = np.zeros(ndim, dtype=FLOAT)
        self.upper_bounds = np.ones(ndim, dtype=FLOAT)
        self.integrality = np.zeros(ndim, dtype=bool)
        self.boolean_mask = np.zeros(ndim, dtype=bool)
        self.known_optimum_point = np.full(ndim, 0.9, dtype=FLOAT)
        self.known_optimum_value = -0.43 if maximize else 2.43
        self.maximize = maximize

    def re__init__(self, **kwargs):
        if 'ndim' in kwargs.keys():
            self.ndim = kwargs['ndim']
            self.lower_bounds = np.zeros(self.ndim, dtype=FLOAT)
            self.upper_bounds = np.ones(self.ndim, dtype=FLOAT)
            self.integrality = np.zeros(self.ndim, dtype=bool)
            self.boolean_mask = np.zeros(self.ndim, dtype=bool)
            self.known_optimum_point = np.full(self.ndim, 0.9, dtype=FLOAT)
        if 'maximize' in kwargs.keys():
            self.maximize = kwargs['maximize']
            self.known_optimum_value = -0.43 if self.maximize else 2.43
        

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

def test_populate_randomly_slice(mock_problem, rng):
    """Tests the _populate_randomly helper function on a slice of niches."""
    points = np.zeros((3, 5, mock_problem.ndim))
    pp._populate_randomly(points, 1, 3, mock_problem.lower_bounds, mock_problem.upper_bounds, rng)
    # Niche 0 should remain untouched
    assert np.all(points[0] == 0)
    # Niches 1 and 2 should be populated
    assert np.any(points[1] != 0)
    assert np.any(points[2] != 0)
    assert np.all(points[1:] >= mock_problem.lower_bounds)
    assert np.all(points[1:] <= mock_problem.upper_bounds)

# Tests for Population class methods
def test_init(population_instance):
    """Tests the initialization of the Population class."""
    pop = population_instance

    num_niches, pop_size, ndim = 2, 5, 3
    
    assert pop.num_niches == num_niches, f"Expected num_niches to be {num_niches}, but got {pop.num_niches}"
    assert pop.pop_size == pop_size, f"Expected pop_size to be {pop_size}, but got {pop.pop_size}"
    assert pop.problem.ndim == ndim, f"Expected ndim to be {ndim}, but got {pop.problem.ndim}"

    assert pop.points.shape == (num_niches, pop_size, ndim), f"Population.points has wrong shape. Expected: {(num_niches, pop_size, ndim)}, got: {pop.points.shape}"
    assert pop.objective_values.shape == (num_niches, pop_size), f"Population.objective_values has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.objective_values.shape}"
    assert pop.violations.shape == (num_niches, pop_size), f"Population.violations has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.violations.shape}"
    assert pop.penalized_objectives.shape == (num_niches, pop_size), f"Population.penalized_objectives has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.penalized_objectives.shape}"
    assert pop.fitnesses.shape == (num_niches, pop_size), f"Population.fitnesses has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.fitnesses.shape}"
    assert pop.is_noptimal.shape == (num_niches, pop_size), f"Population.is_noptimal has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.is_noptimal.shape}"
    assert pop.centroids.shape == (num_niches, ndim), f"Population.centroids has wrong shape. Expected: {(num_niches, ndim)}, got: {pop.centroids.shape}"
    assert pop.niche_elites.shape == (num_niches - 1, 1, ndim), f"Population.niche_elites has wrong shape. Expected: {(num_niches - 1, 1, ndim)}, got: {pop.niche_elites.shape}"
    assert pop.parents.shape == (num_niches, 0, ndim), f"Population.parents has wrong shape. Expected: {(num_niches, 0, ndim)}, got: {pop.niche_elites.shape}"
    assert pop.current_optima.shape == (num_niches, ndim), f"Population.current_optima has wrong shape. Expected: {(num_niches, ndim)}, got: {pop.current_optima.shape}"
    assert pop.current_optima_obj.shape == (num_niches,), f"Population.current_optima_obj has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_obj.shape}"
    assert pop.current_optima_pob.shape == (num_niches,), f"Population.current_optima_pob has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_pob.shape}"
    assert pop.current_optima_fit.shape == (num_niches,), f"Population.current_optima_fit has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_fit.shape}"
    assert pop.current_optima_nop.shape == (num_niches,), f"Population.current_optima_nop has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_nop.shape}"

def test_evaluate_fitness_simple(population_instance):
    """Tests the fitness evaluation logic with predictable points."""
    pop = population_instance
    ndim = pop.problem.ndim # fewer characters -> convenience
    pop.points = np.array([
        [x*np.ones(ndim) for x in range(5)], # Niche 0 -> Centroid [2, 2, 2]
        [x*np.ones(ndim) for x in range(3, 8)],  # Niche 1 -> Centroid [5, 5, 5]
    ], dtype=FLOAT)
    
    pop._evaluate_fitness()
    
    assert (pop.centroids[0] == 2*np.ones(ndim)).all()
    assert (pop.centroids[1] == 5*np.ones(ndim)).all()

    # Expected fitness for niche 0 points is distance to niche 1 centroid
    for i in range(pop.num_niches):
        for j in range(pop.pop_size):
            assert np.isclose(pop.fitnesses[i, j], np.sqrt(((pop.points[i, j] - pop.centroids[i-1])**2).sum()))

def test_evaluate_fitness_3d(population_instance):
    """Tests the fitness evaluation logic with predictable points."""
    pop = population_instance
    ndim = pop.problem.ndim # fewer characters -> convenience
    pop.add_niches(1)
    pop.points = np.array([
        [x*np.ones(ndim) for x in range(5)], # Niche 0 -> Centroid [2, 2, 2]
        [x*np.ones(ndim) for x in range(3, 8)],  # Niche 1 -> Centroid [5, 5, 5]
        [x*np.ones(ndim) for x in range(6, 11)],  # Niche 1 -> Centroid [8, 8, 8]
    ], dtype=FLOAT)
    
    pop._evaluate_fitness()
    
    assert (pop.centroids[0] == 2*np.ones(ndim)).all()
    assert (pop.centroids[1] == 5*np.ones(ndim)).all()
    assert (pop.centroids[2] == 8*np.ones(ndim)).all()

    # Expected fitness for niche 0 points is distance to niche 1 centroid
    for i in range(pop.num_niches):
        idxs = {0:[1, 2], 1:[0, 2], 2:[0, 1]}[i]
        for j in range(pop.pop_size):
            distances = [np.sqrt(((pop.points[i, j] - pop.centroids[i_])**2).sum()) for i_ in idxs]
            # select euclidean distance to closest other centroid
            assert np.isclose(pop.fitnesses[i, j], min(distances))

def test_populate(population_instance):
    """Tests random population generation."""
    pop = population_instance
    pop.populate(np.inf, 0.0)
    assert np.all(pop.points >= pop.problem.lower_bounds)
    assert np.all(pop.points <= pop.problem.upper_bounds)

def test_add_niches(population_instance):
    """Tests adding new niches to the population."""
    pop = population_instance
    initial_niches = pop.num_niches
    pop.add_niches(3)
    assert pop.num_niches == initial_niches + 3
    assert pop.points.shape[0] == initial_niches + 3
    with pytest.raises(ValueError):
        pop.add_niches(-1)

def test_resize_niche_size(population_instance):
    """Tests resizing the population size of each niche."""
    pop = population_instance

    pop._resize_niche_size(5)
    num_niches = 5

    working_arrays = ("points", "objective_values", "violations", 
                      "penalized_objectives", "fitnesses", "is_noptimal",
                      "centroids", "parents", "current_optima",
                      "current_optima_obj", "current_optima_pob", "current_optima_fit", 
                      "current_optima_nop")
    
    for array in working_arrays:
        observed = getattr(pop, array).shape[0]
        assert observed == num_niches, f"Population array: {array} wrong number of niches."\
            f"Expected: {num_niches}, got: {observed}."
    assert pop.niche_elites.shape[0] == num_niches-1, "Population array: niche_elites wrong number of niches."\
            f"Expected: {num_niches-1}, got: {pop.niche_elites.shape[0]}."
    
    with pytest.raises(Exception):
        # cannot decrease
        pop._resize_niche_size(3)
    # after failed resize, arrays have not been resized. 
    # this is not actually a big deal but sometime you might run into this in interactive kernel
    for array in working_arrays:
        observed = getattr(pop, array).shape[0]
        assert observed == num_niches, f"Population array: {array} wrong number of niches."\
            f"Expected: {num_niches}, got: {observed}."
    assert pop.niche_elites.shape[0] == num_niches-1, "Population array: niche_elites wrong number of niches."\
            f"Expected: {num_niches-1}, got: {pop.niche_elites.shape[0]}."


    pop._resize_niche_size(6)
    num_niches = 6
    for array in working_arrays:
        observed = getattr(pop, array).shape[0]
        assert observed == num_niches, f"Population array: {array} wrong number of niches."\
            f"Expected: {num_niches}, got: {observed}."
    assert pop.niche_elites.shape[0] == num_niches-1, "Population array: niche_elites wrong number of niches."\
            f"Expected: {num_niches-1}, got: {pop.niche_elites.shape[0]}."


def test_resize_parent_size(population_instance):
    """Tests resizing the population size of each niche."""
    pop = population_instance

    assert pop.parents.shape == (2, 0, 3)
    assert pop.parent_size == 0
    assert pop.parents.dtype == FLOAT

    pop._resize_parent_size(5)
    assert pop.parents.shape == (2, 5, 3)
    assert pop.parent_size == 5
    assert pop.parents.dtype == FLOAT

    pop._resize_parent_size(10)
    assert pop.parents.shape == (2, 10, 3)
    assert pop.parent_size == 10
    assert pop.parents.dtype == FLOAT

def test_resize_pop_size(population_instance):
    """Tests resizing the population size of each niche."""
    pop = population_instance
    pop.populate(np.inf, 0.0)
    
    # Increase size
    pop.resize(pop_size=10)
    assert pop.pop_size == 10
    assert pop.points.shape == (2, 10, 3)

    # Decrease size
    # internal values needed for resizing down
    pop._resize_parent_size(5)
    pop._evaluate_and_update(1.1, 1.0)
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
    pop.problem.re__init__(maximize=False)
    pop.current_optima_obj[0] = pop.problem.known_optimum_value
    pop.populate(np.inf, 0.0)
    
    # Manually set a point to be better than the known optimum
    pop.points[niche_idx, 3, :] = np.array([0.01, 0.01, 0.01])
    initial_optimum = pop.current_optima_obj[0]
    
    pop._evaluate_and_update(noptimal_slack=1.1, violation_factor=1.0)
    
    # New optimum should be sum of squares of [0.01, 0.01, 0.01] = 0.0003
    assert pop.current_optima_obj[0] <= initial_optimum
    assert np.isclose(pop.current_optima_obj[0], 0.0003)
    assert np.allclose(pop.current_optima[0], [0.01, 0.01, 0.01])

@pytest.mark.parametrize("niche_idx", [0, 1])
def test_update_optima_maximize(population_instance, niche_idx):
    """Tests if the global optimum is updated correctly for maximization."""
    pop = population_instance
    pop.problem.re__init__(maximize=True)
    pop.current_optima_obj[0] = pop.problem.known_optimum_value
    pop.populate(np.inf, 0.0)

    # Manually set a point with a high objective value (close to zero)
    pop.points[niche_idx, 3, :] = np.array([0.01, 0.01, 0.01])
    initial_optimum = pop.current_optima_obj[0] 
    
    pop._evaluate_and_update(noptimal_slack=1.1, violation_factor=1.0)
    
    # New optimum should be offset -sum of squares of [0.01, 0.01, 0.01] = 2-0.0003
    assert pop.current_optima_obj[0] >= initial_optimum
    assert np.isclose(pop.current_optima_obj[0], 2-0.0003)
    assert np.allclose(pop.current_optima[0], [0.01, 0.01, 0.01])

def test_evolve(population_instance):
    """Tests a single evolution step."""
    pop = population_instance
    pop.populate(np.inf, 1.0)
    
    initial_points = pop.points.copy()
    pop.resize(parent_size=3)

    num_niches = pop.num_niches
    pop_size = pop.pop_size
    ndim = pop.problem.ndim
    parent_size = pop.parent_size
    current_optimum = pop.current_optima[0].copy()
    pop.evolve(
        elite_count=1,
        tourn_count=2,
        tourn_size=2,
        mutation_prob=1.0,
        mutation_sigma=0.5,
        crossover_prob=0.8,
        niche_elitism="selfish",
        noptimal_slack=np.inf, 
        violation_factor=0.0,
    )
    
    # The points should have changed after evolution
    assert not np.array_equal(initial_points, pop.points)
    # The best individual (optimum) should be preserved
    # (Although there may now be a better optimum)
    assert np.allclose(pop.points[0, 0, :], current_optimum)

    assert pop.points.shape == (num_niches, pop_size, ndim), f"Population.points has wrong shape. Expected: {(num_niches, pop_size, ndim)}, got: {pop.points.shape}"
    assert pop.objective_values.shape == (num_niches, pop_size), f"Population.objective_values has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.objective_values.shape}"
    assert pop.violations.shape == (num_niches, pop_size), f"Population.violations has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.violations.shape}"
    assert pop.penalized_objectives.shape == (num_niches, pop_size), f"Population.penalized_objectives has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.penalized_objectives.shape}"
    assert pop.fitnesses.shape == (num_niches, pop_size), f"Population.fitnesses has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.fitnesses.shape}"
    assert pop.is_noptimal.shape == (num_niches, pop_size), f"Population.is_noptimal has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.is_noptimal.shape}"
    assert pop.centroids.shape == (num_niches, ndim), f"Population.centroids has wrong shape. Expected: {(num_niches, ndim)}, got: {pop.centroids.shape}"
    assert pop.niche_elites.shape == (num_niches - 1, 1, ndim), f"Population.niche_elites has wrong shape. Expected: {(num_niches - 1, 1, ndim)}, got: {pop.niche_elites.shape}"
    assert pop.parents.shape == (num_niches, parent_size, ndim), f"Population.parents has wrong shape. Expected: {(num_niches, 0, ndim)}, got: {pop.niche_elites.shape}"
    assert pop.current_optima.shape == (num_niches, ndim), f"Population.current_optima has wrong shape. Expected: {(num_niches, ndim)}, got: {pop.current_optima.shape}"
    assert pop.current_optima_obj.shape == (num_niches,), f"Population.current_optima_obj has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_obj.shape}"
    assert pop.current_optima_pob.shape == (num_niches,), f"Population.current_optima_pob has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_pob.shape}"
    assert pop.current_optima_fit.shape == (num_niches,), f"Population.current_optima_fit has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_fit.shape}"
    assert pop.current_optima_nop.shape == (num_niches,), f"Population.current_optima_nop has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_nop.shape}"

@pytest.mark.parametrize("niche_elitism", [None, "selfish"])
def test_generate_offspring(niche_elitism, population_instance):
    pop = population_instance
    # setup
    pop.populate(np.inf, 0)
    current_optima = pop.current_optima.copy() # selfish niche elites 
    points = pop.points.copy() 
    pop.elite_count=1
    pop.tourn_count=2
    pop.tourn_size=2
    pop.mutation_prob=0.0 # no mutation
    pop.mutation_sigma=0.0 # no mutation
    pop.crossover_prob=0.0 # no crossover
    pop.niche_elitism=niche_elitism
    pop.resize(parent_size=3)
    pop._select_parents()
    parents = pop.parents.copy().reshape(-1, 3)
    
    # execution
    pop._generate_offspring()

    # validation
    if niche_elitism is None:
        pool = np.vstack((parents, current_optima[0])) # include optimum
        for i in range(pop.num_niches):
            for j in range(pop.pop_size):
                # no mutation + no crossover + no niche elitism + elite_count > 0
                #       ->  points should be in parents 
                assert pop.points[i, j] in parents
    elif niche_elitism == "selfish":
        # selfish elitism ->  preserve individual with highest fitness in each niche
        for i in range(pop.num_niches):
            pool = np.vstack((parents, current_optima[i]))
            for j in range(pop.pop_size):
                # no mutation + no crossover + elite_count > 0 
                #       ->  points should be in parents + niche_elites
                assert pop.points[i, j] in pool
            assert current_optima[i] in pop.points[i]

def test_unselfish_niche_elitism(population_instance):
    pop = population_instance
    # setup
    pop.populate(np.inf, 0)
    pop.elite_count=1
    pop.tourn_count=2
    pop.tourn_size=2
    pop.mutation_prob=0.50 
    pop.mutation_sigma=0.1 
    pop.crossover_prob=0.0 # no crossover
    pop.niche_elitism="unselfish"
    pop.resize(parent_size=3)
    pop._select_parents()
    
    # execution
    unselfish_niche_fit = -1
    for _ in range(50):
        pop._evaluate_and_update(np.inf, 0)
        pop._select_parents()

        pop._generate_offspring()
        assert pop.unselfish_niche_fit >= unselfish_niche_fit
        if pop.unselfish_niche_fit > unselfish_niche_fit:
            unselfish_niche_fit = pop.unselfish_niche_fit
            niche_elites = pop.niche_elites.copy()

        assert pop.current_optima[0] in pop.points[0]
        assert niche_elites[0] in pop.points[1]
