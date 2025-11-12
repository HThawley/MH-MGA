import pytest
import numpy as np

from mga.commons.types import DEFAULTS

INT, FLOAT = DEFAULTS
import mga.population as pp  # noqa: E402


# Tests for @njit helper functions
@pytest.mark.parametrize("ndim", [0, 1, 2, 3])
def test_njit_deepcopy(rng, ndim):
    old = rng.random((10,) * ndim)
    new = np.empty_like(old)
    pp.njit_deepcopy(new, old)
    assert np.array_equal(old, new), "copy not performed correctly"
    if old.size > 0:
        assert id(old) != id(new), "copy not deep"


@pytest.mark.parametrize(
    "optimal_obj, slack, maximize, expected",
    [
        (100.0, 1.1, False, 110.0),
        (100.0, 1.1, True, 90.0),
        (50.0, 1.5, False, 75.0),
        (50.0, 1.5, True, 25.0),
    ],
)
def test_noptimal_threshold(optimal_obj, slack, maximize, expected):
    """Tests the near-optimal threshold calculation."""
    result = pp._noptimal_threshold(optimal_obj, slack, maximize)
    assert np.isclose(result, expected)


def test_clone(rng):
    """Tests the cloning of individuals into a larger population."""
    start_pop = rng.random((2, 3, 4))  # niches, pop_size, ndim
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
    points = np.array(
        [
            [[1, 2, 3], [3, 4, 5]],  # Niche 1
            [[5, 6, 7], [7, 8, 9]],  # Niche 2
        ],
        dtype=FLOAT,
    )
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
    points = rng.random((2, 5, 3)) * 2 - 0.5  # Values between -0.5 and 1.5
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
    assert pop.ndim == ndim, f"Expected ndim to be {ndim}, but got {pop.ndim}"

    assert pop.points.shape == (
        num_niches,
        pop_size,
        ndim,
    ), f"Population.points has bad shape. Expected: {(num_niches, pop_size, ndim)}, got: {pop.points.shape}"
    assert pop.objective_values.shape == (
        num_niches,
        pop_size,
    ), f"Population.objective_values has bad shape. Expected: {(num_niches, pop_size)}, "\
        f"got: {pop.objective_values.shape}"
    assert pop.violations.shape == (
        num_niches,
        pop_size,
    ), f"Population.violations has bad shape. Expected: {(num_niches, pop_size)}, got: {pop.violations.shape}"
    assert pop.penalized_objectives.shape == (
        num_niches,
        pop_size,
    ), f"Population.penalized_objectives has bad shape. Expected: {(num_niches, pop_size)}, "\
        f"got: {pop.penalized_objectives.shape}"
    assert pop.fitnesses.shape == (
        num_niches,
        pop_size,
    ), f"Population.fitnesses has bad shape. Expected: {(num_niches, pop_size)}, "\
        f"got: {pop.fitnesses.shape}"
    assert pop.is_noptimal.shape == (
        num_niches,
        pop_size,
    ), f"Population.is_noptimal has bad shape. Expected: {(num_niches, pop_size)}, got: {pop.is_noptimal.shape}"
    assert pop.centroids.shape == (
        num_niches,
        ndim,
    ), f"Population.centroids has bad shape. Expected: {(num_niches, ndim)}, got: {pop.centroids.shape}"
    assert pop.niche_elites.shape == (
        num_niches - 1,
        1,
        ndim,
    ), f"Population.niche_elites has bad shape. Expected: {(num_niches - 1, 1, ndim)}, got: {pop.niche_elites.shape}"
    assert pop.parents.shape == (
        num_niches,
        0,
        ndim,
    ), f"Population.parents has bad shape. Expected: {(num_niches, 0, ndim)}, got: {pop.niche_elites.shape}"
    assert pop.current_optima.shape == (
        num_niches,
        ndim,
    ), f"Population.current_optima has bad shape. Expected: {(num_niches, ndim)}, got: {pop.current_optima.shape}"
    assert pop.current_optima_obj.shape == (
        num_niches,
    ), f"Population.current_optima_obj has bad shape. Expected: {(num_niches,)}, got: {pop.current_optima_obj.shape}"
    assert pop.current_optima_pob.shape == (
        num_niches,
    ), f"Population.current_optima_pob has bad shape. Expected: {(num_niches,)}, got: {pop.current_optima_pob.shape}"
    assert pop.current_optima_fit.shape == (
        num_niches,
    ), f"Population.current_optima_fit has bad shape. Expected: {(num_niches,)}, got: {pop.current_optima_fit.shape}"
    assert pop.current_optima_nop.shape == (
        num_niches,
    ), f"Population.current_optima_nop has bad shape. Expected: {(num_niches,)}, got: {pop.current_optima_nop.shape}"


def test_evaluate_fitness_simple(population_instance):
    """Tests the fitness evaluation logic with predictable points."""
    pop = population_instance
    ndim = pop.ndim  # fewer characters -> convenience
    pop.points = np.array(
        [
            [x * np.ones(ndim) for x in range(5)],  # Niche 0 -> Centroid [2, 2, 2]
            [x * np.ones(ndim) for x in range(3, 8)],  # Niche 1 -> Centroid [5, 5, 5]
        ],
        dtype=FLOAT,
    )

    pop.evaluate_fitness()

    assert (pop.centroids[0] == 2 * np.ones(ndim)).all()
    assert (pop.centroids[1] == 5 * np.ones(ndim)).all()

    # Expected fitness for niche 0 points is distance to niche 1 centroid
    for i in range(pop.num_niches):
        for j in range(pop.pop_size):
            assert np.isclose(pop.fitnesses[i, j], np.sqrt(((pop.points[i, j] - pop.centroids[i - 1]) ** 2).sum()))


def test_evaluate_fitness_3d(population_instance):
    """Tests the fitness evaluation logic with predictable points."""
    pop = population_instance
    ndim = pop.ndim  # fewer characters -> convenience
    pop.add_niches(1)
    pop.points = np.array(
        [
            [x * np.ones(ndim) for x in range(5)],  # Niche 0 -> Centroid [2, 2, 2]
            [x * np.ones(ndim) for x in range(3, 8)],  # Niche 1 -> Centroid [5, 5, 5]
            [x * np.ones(ndim) for x in range(6, 11)],  # Niche 1 -> Centroid [8, 8, 8]
        ],
        dtype=FLOAT,
    )

    pop.evaluate_fitness()

    assert (pop.centroids[0] == 2 * np.ones(ndim)).all()
    assert (pop.centroids[1] == 5 * np.ones(ndim)).all()
    assert (pop.centroids[2] == 8 * np.ones(ndim)).all()

    # Expected fitness for niche 0 points is distance to niche 1 centroid
    for i in range(pop.num_niches):
        idxs = {0: [1, 2], 1: [0, 2], 2: [0, 1]}[i]
        for j in range(pop.pop_size):
            distances = [np.sqrt(((pop.points[i, j] - pop.centroids[i_]) ** 2).sum()) for i_ in idxs]
            # select euclidean distance to closest other centroid
            assert np.isclose(pop.fitnesses[i, j], min(distances))


def test_populate(population_instance):
    """Tests random population generation."""
    pop = population_instance
    pop.populate(np.inf, 0.0)
    assert np.all(pop.points >= pop.lower_bounds)
    assert np.all(pop.points <= pop.upper_bounds)


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

    working_arrays = (
        "points",
        "objective_values",
        "violations",
        "penalized_objectives",
        "fitnesses",
        "is_noptimal",
        "centroids",
        "parents",
        "current_optima",
        "current_optima_obj",
        "current_optima_pob",
        "current_optima_fit",
        "current_optima_nop",
    )

    for array in working_arrays:
        observed = getattr(pop, array).shape[0]
        assert observed == num_niches, (
            f"Population array: {array} wrong number of niches." f"Expected: {num_niches}, got: {observed}."
        )
    assert pop.niche_elites.shape[0] == num_niches - 1, (
        "Population array: niche_elites wrong number of niches."
        f"Expected: {num_niches-1}, got: {pop.niche_elites.shape[0]}."
    )

    with pytest.raises(Exception):
        # cannot decrease
        pop._resize_niche_size(3)
    # after failed resize, arrays have not been resized.
    # this is not actually a big deal but sometime you might run into this in interactive kernel
    for array in working_arrays:
        observed = getattr(pop, array).shape[0]
        assert observed == num_niches, (
            f"Population array: {array} wrong number of niches." f"Expected: {num_niches}, got: {observed}."
        )
    assert pop.niche_elites.shape[0] == num_niches - 1, (
        "Population array: niche_elites wrong number of niches."
        f"Expected: {num_niches-1}, got: {pop.niche_elites.shape[0]}."
    )

    pop._resize_niche_size(6)
    num_niches = 6
    for array in working_arrays:
        observed = getattr(pop, array).shape[0]
        assert observed == num_niches, (
            f"Population array: {array} wrong number of niches." f"Expected: {num_niches}, got: {observed}."
        )
    assert pop.niche_elites.shape[0] == num_niches - 1, (
        "Population array: niche_elites wrong number of niches."
        f"Expected: {num_niches-1}, got: {pop.niche_elites.shape[0]}."
    )


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


def test_resize_pop_size(population_instance, mock_problem):
    """Tests resizing the population size of each niche."""
    pop = population_instance
    pop.populate()

    # Increase size
    pop.resize(-1, 10, -1)
    assert pop.pop_size == 10
    assert pop.points.shape == (2, 10, 3)

    pop.noptimal_slack = 1.1
    pop.violation_factor = 1.0

    # Decrease size
    # internal values needed for resizing down
    pop._resize_parent_size(5)
    mock_problem.evaluate_population(pop)
    pop.evaluate_fitness()
    pop.update_optima()
    pop.elite_count, pop.tourn_count, pop.tourn_size = 5, 0, 0
    pop.resize(-1, 4, -1)
    assert pop.pop_size == 4
    assert pop.points.shape == (2, 4, 3)


def test_apply_integrality(population_instance):
    """Tests the enforcement of integrality constraints."""
    pop = population_instance
    pop.integrality[1] = True  # Make the second dimension integer
    pop.points[:] = 0.4
    pop._apply_integrality()
    assert np.all(pop.points[:, :, 0] == 0.4)
    assert np.all(pop.points[:, :, 1] == 0.0)  # Should be rounded
    assert np.all(pop.points[:, :, 2] == 0.4)


@pytest.mark.parametrize("niche_elitism_int", [0, 1])
def test_generate_offspring(niche_elitism_int, population_instance):
    pop = population_instance
    # setup
    pop.populate()
    current_optima = pop.current_optima.copy()  # selfish niche elites
    # points = pop.points.copy()
    pop.elite_count = 1
    pop.tourn_count = 2
    pop.tourn_size = 2
    pop.mutation_prob = np.array([0.0, 0.0])  # no mutation
    pop.mutation_sigma = np.array([0.0, 0.0])  # no mutation
    pop.crossover_prob = np.array([0.0, 0.0])  # no crossover
    pop.niche_elitism = niche_elitism_int
    pop.resize(-1, -1, 3)
    pop.select_parents()
    parents = pop.parents.copy().reshape(-1, 3)

    # execution
    pop.generate_offspring()

    # validation
    if niche_elitism_int == 0:
        pool = np.vstack((parents, current_optima[0]))  # include optimum
        for i in range(pop.num_niches):
            for j in range(pop.pop_size):
                # no mutation + no crossover + no niche elitism + elite_count > 0
                #       ->  points should be in parents
                assert pop.points[i, j] in parents, "offspring not drawn from parents"
    elif niche_elitism_int == 1:
        # selfish elitism ->  preserve individual with highest fitness in each niche
        for i in range(pop.num_niches):
            pool = np.vstack((parents, current_optima[i]))
            for j in range(pop.pop_size):
                # no mutation + no crossover + elite_count > 0
                #       ->  points should be in parents + niche_elites
                assert pop.points[i, j] in pool, "offspring not drawn from parents"
            assert current_optima[i] in pop.points[i], "selfish elitishm did not preserve optima"


def test_unselfish_niche_elitism(population_instance, mock_problem):
    pop = population_instance
    # setup
    pop.populate(np.inf, 0)
    pop.elite_count = 1
    pop.tourn_count = 2
    pop.tourn_size = 2
    pop.mutation_prob = np.array([0.5, 0.5])
    pop.mutation_sigma = np.array([0.1, 0.1])
    pop.crossover_prob = np.array([0.0, 0.0])
    pop.niche_elitism = 2
    pop.resize(-1, -1, 3)
    pop.select_parents()

    # execution
    unselfish_niche_fit = -1
    for _ in range(50):
        mock_problem.evaluate_population(pop)
        pop.evaluate_fitness()
        pop.update_optima()
        pop.select_parents()

        pop.generate_offspring()
        assert pop.unselfish_niche_fit >= unselfish_niche_fit
        if pop.unselfish_niche_fit > unselfish_niche_fit:
            unselfish_niche_fit = pop.unselfish_niche_fit
            niche_elites = pop.niche_elites.copy()

        assert pop.current_optima[0] in pop.points[0]
        assert niche_elites[0] in pop.points[1]
