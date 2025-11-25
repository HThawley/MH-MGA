import pytest
import numpy as np
import shutil
import os
from datetime import datetime

from mga.commons.types import DEFAULTS

FLOAT, INT = DEFAULTS
from mga.mhmga import MGAProblem  # noqa: E402
from mga.utils.logger import Logger  # noqa: E402
from mga.utils.typing import is_boolean, is_integer, is_float  # noqa: E402
from mga.population import load_problem_to_population  # noqa: E402
import mga.population as population_module  # noqa: E402


# Fixtures
class MockPopulation:
    """A mock Population class that allows spying on method calls."""

    def __init__(self, num_niches, pop_size, ndim, rng, stable_sort):
        self.num_niches = num_niches
        self.pop_size = pop_size
        self.rng = rng
        self.ndim = ndim
        self.stable_sort = stable_sort
        self.current_optima = rng.random((num_niches, ndim))
        self.current_optima_fit = rng.random(num_niches)
        self.current_optima_obj = rng.random(num_niches)
        self.current_optima_nop = np.ones(num_niches, dtype=bool)
        self.mean_fitness = 0
        self.current_best_obj = 0

        # spying
        self.num_niches_called = 0
        self.add_niches_calls = 0

    def populate(self, *args, **kwargs):
        pass

    def resize(self, *args, **kwargs):
        pass

    def add_niches(self, num_niches):
        self.num_niches_called = num_niches
        if hasattr(self, "add_niches_calls"):
            self.add_niches_calls += 1
        else:
            self.add_niches_calls = 1


# Tests
def test_mga_init(mga_instance):
    """Tests the initialization of the MGAProblem class and attribute types."""
    # Assert that all attributes are initialized with the correct types
    # Assert that all attributes are initialized with the correct types
    assert isinstance(
        mga_instance.rng, np.random.Generator
    ), f"'rng' has bad type. Expected: np.random.Generator, got: {type(mga_instance.rng)}"
    assert is_boolean(
        mga_instance.stable_sort
    ), f"'stable_sort' has bad type. Expected: bool, got: {type(mga_instance.stable_sort)}"
    assert mga_instance.logger is None, "logger should be None when no log_dir is provided"
    assert mga_instance.population is None, "population should be None at initialization"
    assert is_integer(
        mga_instance.current_iter
    ), f"'current_iter' has bad type. Expected: int, got: {type(mga_instance.current_iter)}"
    assert isinstance(
        mga_instance.start_time, datetime
    ), f"'start_time' has bad type. Expected: datetime, got: {type(mga_instance.start_time)}"
    assert is_boolean(
        mga_instance._is_populated
    ), f"'_is_populated' has bad type. Expected: bool, got: {type(mga_instance._is_populated)}"
    assert is_integer(
        mga_instance.pop_size
    ), f"'pop_size' has bad type. Expected: int, got: {type(mga_instance.pop_size)}"
    assert is_integer(
        mga_instance.elite_count
    ), f"'elite_count' has bad type. Expected: int, got: {type(mga_instance.elite_count)}"
    assert is_integer(
        mga_instance.tourn_count
    ), f"'tourn_count' has bad type. Expected: int got: {type(mga_instance.tourn_count)}"
    assert is_integer(
        mga_instance.tourn_size
    ), f"'tourn_size' has bad type. Expected: int, got: {type(mga_instance.tourn_size)}"
    assert is_float(
        mga_instance.mutation_prob
    ), f"'mutation_prob' has bad type. Expected: float, got: {type(mga_instance.mutation_prob)}"
    assert is_float(
        mga_instance.mutation_sigma
    ), f"'mutation_sigma' has bad type. Expected: float, got: {type(mga_instance.mutation_sigma)}"
    assert is_float(
        mga_instance.crossover_prob
    ), f"'crossover_prob' has bad type. Expected: float, got: {type(mga_instance.crossover_prob)}"
    assert mga_instance.niche_elitism is None, "niche_elitism should be None at initialization"
    assert is_float(
        mga_instance.noptimal_slack
    ), f"'noptimal_slack' has bad type. Expected: float, got: {type(mga_instance.noptimal_slack)}"


def test_mga_init_with_logger(mock_problem):
    """Tests initialization with a logger directory and ensures cleanup."""
    # Ensure the directory does not exist before the test
    if os.path.exists("test_logs"):
        os.chmod("test_logs", 0o666)
        shutil.rmtree("test_logs", ignore_errors=True)
    try:
        mga_prob = MGAProblem(problem=mock_problem, log_dir="test_logs/log", log_freq=1)
        assert isinstance(mga_prob.logger, Logger), f"'MGAProblem.logger' is not a Logger, Got: {type(mga_prob.logger)}"
        assert mga_prob.logger is not None, "Logger should be instantiated when log_dir is provided"
        assert os.path.exists("test_logs"), "Log directory should be created"
    finally:
        # This block will run whether the test fails or succeeds
        if os.path.exists("test_logs"):
            os.chmod("test_logs", 0o666)
            shutil.rmtree("test_logs")


def test_add_niches_calls_population_method(mga_instance, monkeypatch):
    """Tests that MGAProblem.add_niches correctly calls Population.add_niches."""
    monkeypatch.setattr(population_module, "Population", MockPopulation)

    mga_instance.pop_size = 5
    mga_instance.violation_factor = 0.0
    mga_instance.noptimal_slack = np.inf

    mga_instance.add_niches(5)
    assert mga_instance.num_niches == 5, "num_niches should be set before population is created"
    assert mga_instance.population is None, "population should not initiated until explicitly called"
    # Simulate population creation
    mga_instance.populate()
    # Call add_niches again
    mga_instance.add_niches(3)
    # Verify that the population's add_niches method was called with the correct argument
    assert mga_instance.population.num_niches_called == 3, "'num_niches' not passed correctly to 'Population'"
    assert mga_instance.population.add_niches_calls == 1, "'mga.add_niches' did not call 'mga.Population.add_niches'"
    assert mga_instance.num_niches == 8, "Total niches should be updated after adding more"


def test_step_runs(mga_instance):
    """Tests that the main step function runs for a few iterations without errors."""
    mga_instance.add_niches(2)
    mga_instance.update_hyperparameters(max_iter=5)
    mga_instance.step()
    assert mga_instance.current_iter == 5, "The algorithm should run for the specified number of iterations"


def test_hyperparameter_input_validation(mga_instance):
    """Tests the comprehensive input validation for hyperparameters ."""
    mga_instance.add_niches(2)

    # Test max_iter
    with pytest.raises(TypeError, match="max_iter"):
        mga_instance.update_hyperparameters(max_iter=5.5)
    with pytest.raises(ValueError, match="max_iter"):
        mga_instance.update_hyperparameters(max_iter=0)

    # Test pop_size
    with pytest.raises(TypeError, match="pop_size"):
        mga_instance.update_hyperparameters(max_iter=5, pop_size=100.5)
    with pytest.raises(ValueError, match="pop_size"):
        mga_instance.update_hyperparameters(max_iter=5, pop_size=0)

    # Test elite_count
    with pytest.raises(TypeError, match="elite_count"):
        mga_instance.update_hyperparameters(max_iter=5, elite_count="a")
    with pytest.raises(ValueError, match="elite_count"):
        mga_instance.update_hyperparameters(max_iter=5, elite_count=1.1)

    # Test tourn_count
    with pytest.raises(TypeError, match="tourn_count"):
        mga_instance.update_hyperparameters(max_iter=5, tourn_count=None)
    with pytest.raises(ValueError, match="tourn_count"):
        mga_instance.update_hyperparameters(max_iter=5, tourn_count=-0.1)

    # Test tourn_size
    with pytest.raises(TypeError, match="tourn_size"):
        mga_instance.update_hyperparameters(max_iter=5, tourn_size=2.5)
    with pytest.raises(ValueError, match="tourn_size"):
        mga_instance.update_hyperparameters(max_iter=5, tourn_size=0)

    # Test niche_elitism
    with pytest.raises(ValueError, match="niche_elitism"):
        mga_instance.update_hyperparameters(max_iter=5, niche_elitism="invalid_option")

    # Test multi-input constraints
    with pytest.raises(ValueError, match="elite_count \\+ tourn_count \\+ pop_size"):
        mga_instance.update_hyperparameters(max_iter=5, pop_size=100, elite_count=50, tourn_count=60)
    with pytest.raises(ValueError, match="tourn_size \\+ pop_size"):
        mga_instance.update_hyperparameters(max_iter=5, pop_size=10, tourn_size=11)


def test_populate(mga_instance):
    """Tests that the population is created and initialized correctly."""
    mga_instance.add_niches(3)
    assert not mga_instance._is_populated, "Should not be populated before calling populate"
    mga_instance.update_hyperparameters(max_iter=5, pop_size=20)
    mga_instance.populate()

    assert mga_instance._is_populated, "Should be populated after calling populate"
    assert mga_instance.population is not None, "Population object should be created"
    assert mga_instance.population.pop_size == 20, "Population size should be set correctly"


def test_get_results(mga_instance):
    """Tests that get_results returns the correct objects from the Population instance."""
    with pytest.raises(RuntimeError, match="Algorithm has not been run yet."):
        mga_instance.get_results()

    mga_instance.add_niches(2)
    mga_instance.update_hyperparameters(max_iter=1)
    mga_instance.step()

    results = mga_instance.get_results()

    assert (
        results["optima"] == mga_instance.population.current_optima
    ).all(), "Results 'optima' should be the same object as population's optima"
    assert (
        results["fitness"] == mga_instance.population.current_optima_fit
    ).all(), "Results 'fitness' should be the same object as population's fitness"
    assert (
        results["objective"] == mga_instance.population.current_optima_obj
    ).all(), "Results 'objective' should be the same object as population's objective"
    assert (
        results["penalties"] == (mga_instance.population.current_optima_pob
                                 - mga_instance.population.current_optima_obj)
    ).all(), "Results 'penalties' not properly calculated"
    assert (
        results["noptimality"] == mga_instance.population.current_optima_nop
    ).all(), "Results 'noptimality' should be the same object as population's noptimality"


@pytest.mark.parametrize("niche_idx", [0, 1])
def test_update_optima_minimize(population_instance, mock_problem, niche_idx):
    """Tests if the global optimum is updated correctly for minimization."""
    pop = population_instance
    mock_problem.re__init__(maximize=False)
    load_problem_to_population(pop, mock_problem)
    pop.current_optima_obj[0] = mock_problem.known_optimum_value
    pop.populate()

    pop.noptimal_slack = 1.1
    pop.violation_factor = 1.0

    # Manually set a point to be better than the known optimum
    pop.points[niche_idx, 3, :] = np.array([0.01, 0.01, 0.01])
    initial_optimum = pop.current_optima_obj[0]

    mock_problem.evaluate_population(pop)
    pop.evaluate_fitness()
    pop.update_optima()

    # New optimum should be sum of squares of [0.01, 0.01, 0.01] = 0.0003
    assert pop.current_optima_obj[0] <= initial_optimum
    assert np.isclose(pop.current_optima_obj[0], 0.0003)
    assert np.allclose(pop.current_optima[0], [0.01, 0.01, 0.01])


@pytest.mark.parametrize("niche_idx", [0, 1])
def test_update_optima_maximize(population_instance, mock_problem, niche_idx):
    """Tests if the global optimum is updated correctly for maximization."""
    pop = population_instance
    mock_problem.re__init__(maximize=True)
    load_problem_to_population(pop, mock_problem)
    pop.current_optima_obj[0] = mock_problem.known_optimum_value
    pop.populate()

    pop.noptimal_slack = 1.1
    pop.violation_factor = 1.0

    # Manually set a point with a high objective value (close to zero)
    pop.points[niche_idx, 3, :] = np.array([0.01, 0.01, 0.01])
    initial_optimum = pop.current_optima_obj[0]

    mock_problem.evaluate_population(pop)
    pop.evaluate_fitness()
    pop.update_optima()

    # New optimum should be offset -sum of squares of [0.01, 0.01, 0.01] = 2-0.0003
    assert pop.current_optima_obj[0] >= initial_optimum
    assert np.isclose(pop.current_optima_obj[0], 2 - 0.0003)
    assert np.allclose(pop.current_optima[0], [0.01, 0.01, 0.01])


def test_step_evolves_population(mga_instance):
    """Tests a single evolution step."""
    num_niches = 2
    pop_size = 5
    elite_count = 1
    tourn_count = 2
    parent_size = elite_count + tourn_count
    ndim = mga_instance.problem.ndim

    mga_instance.add_niches(num_niches)
    mga_instance.update_hyperparameters(
        max_iter=1,  # Run for a single step
        pop_size=pop_size,
        elite_count=elite_count,
        tourn_count=tourn_count,
        tourn_size=2,
        mutation_prob=1.0,
        mutation_sigma=0.5,
        crossover_prob=0.8,
        niche_elitism="selfish",
        noptimal_slack=np.inf,
        violation_factor=0.0,
    )

    mga_instance.populate()
    mga_instance.problem.evaluate_population(mga_instance.population)
    pop = mga_instance.population

    initial_points = pop.points.copy()
    current_optimum = pop.current_optima[0].copy()

    assert mga_instance.current_iter == 1, "'current_iter' did not increment"

    # The points should have changed after evolution
    assert not np.array_equal(initial_points, pop.points), "Population points did not change after step"
    # The best individual (optimum) should be preserved
    # (Although there may now be a better optimum)
    assert np.allclose(pop.points[0, 0, :], current_optimum), "elitism failed to preserve previous optimum"

    assert pop.points.shape == (
        num_niches,
        pop_size,
        ndim,
    ), f"Population.points has wrong shape. Expected: {(num_niches, pop_size, ndim)}, got: {pop.points.shape}"
    assert pop.objective_values.shape == (
        num_niches,
        pop_size,
    ), f"Population.objective_values has wrong shape. Expected: {(num_niches, pop_size)}, "\
        f"got: {pop.objective_values.shape}"
    assert pop.violations.shape == (
        num_niches,
        pop_size,
    ), f"Population.violations has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.violations.shape}"
    assert pop.penalized_objectives.shape == (
        num_niches,
        pop_size,
    ), f"Population.penalized_objectives has wrong shape. Expected: {(num_niches, pop_size)}, "\
        f"got: {pop.penalized_objectives.shape}"
    assert pop.fitnesses.shape == (
        num_niches,
        pop_size,
    ), f"Population.fitnesses has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.fitnesses.shape}"
    assert pop.is_noptimal.shape == (
        num_niches,
        pop_size,
    ), f"Population.is_noptimal has wrong shape. Expected: {(num_niches, pop_size)}, got: {pop.is_noptimal.shape}"
    assert pop.centroids.shape == (
        num_niches,
        ndim,
    ), f"Population.centroids has wrong shape. Expected: {(num_niches, ndim)}, got: {pop.centroids.shape}"
    assert pop.niche_elites.shape == (
        num_niches - 1,
        1,
        ndim,
    ), f"Population.niche_elites has wrong shape. Expected: {(num_niches - 1, 1, ndim)}, got: {pop.niche_elites.shape}"
    assert pop.parents.shape == (
        num_niches,
        parent_size,
        ndim,
    ), f"Population.parents has wrong shape. Expected: {(num_niches, 0, ndim)}, got: {pop.niche_elites.shape}"
    assert pop.current_optima.shape == (
        num_niches,
        ndim,
    ), f"Population.current_optima has wrong shape. Expected: {(num_niches, ndim)}, got: {pop.current_optima.shape}"
    assert pop.current_optima_obj.shape == (
        num_niches,
    ), f"Population.current_optima_obj has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_obj.shape}"
    assert pop.current_optima_pob.shape == (
        num_niches,
    ), f"Population.current_optima_pob has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_pob.shape}"
    assert pop.current_optima_fit.shape == (
        num_niches,
    ), f"Population.current_optima_fit has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_fit.shape}"
    assert pop.current_optima_nop.shape == (
        num_niches,
    ), f"Population.current_optima_nop has wrong shape. Expected: {(num_niches,)}, got: {pop.current_optima_nop.shape}"
