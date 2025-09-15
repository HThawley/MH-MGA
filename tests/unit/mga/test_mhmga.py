import pytest
import numpy as np
import shutil
import os
from datetime import datetime
from unittest.mock import MagicMock

from mga.commons.types import DEFAULTS
FLOAT, INT = DEFAULTS
from mga.mhmga import MGAProblem
from mga.problem_definition import OptimizationProblem
from mga.population import Population
from mga.utils.logger import Logger
from mga.utils.type_asserts import (is_boolean, is_integer, is_float)

# Mock classes from other modules to isolate testing of MGAProblem
class MockOptimizationProblem(OptimizationProblem):
    """A mock OptimizationProblem class for testing purposes."""
    def __init__(self, ndim=3, maximize=True):
        self.ndim = ndim
        self.lower_bounds = np.zeros(ndim)
        self.upper_bounds = np.ones(ndim)
        self.integrality = np.zeros(ndim, dtype=bool)
        self.boolean_mask = np.zeros(ndim, dtype=bool)
        self.known_optimum_point = np.full(ndim, 0.5)
        self.known_optimum_value = 1.0 if maximize else 0.0
        self.maximize = maximize
        self.rng = np.random.default_rng(42)

    def evaluate(self, points):
        obj = np.sum(points, axis=1)
        vio = np.zeros(points.shape[0])
        return obj, vio

class MockPopulation(Population):
    """A mock Population class that allows spying on method calls."""
    def __init__(self, problem, num_niches, pop_size, rng, stable_sort):
        self.problem = problem
        self.num_niches = num_niches
        self.pop_size = pop_size
        self.rng = rng
        self.stable_sort = stable_sort
        self.current_optima = rng.random((num_niches, problem.ndim))
        self.current_optima_fit = rng.random(num_niches)
        self.current_optima_obj = rng.random(num_niches)
        self.current_optima_nop = np.ones(num_niches, dtype=bool)
        self.mean_fitness=0
        self.current_best_obj=0

    def populate(self, *args, **kwargs):
        pass

    def _evaluate_and_update(self, *args, **kwargs):
        pass

    def evolve(self, *args, **kwargs):
        pass

    def resize(self, *args, **kwargs):
        pass

    def add_niches(self, num_niches):
        self.num_niches_called = num_niches
        if hasattr(self, "add_niches_calls"):
            self.add_niches_calls += 1
        else: 
            self.add_niches_calls = 1


# Monkeypatch the real Population class with our mock
from mga import mhmga
mhmga.Population = MockPopulation

# Fixtures
@pytest.fixture
def mock_problem():
    """Provides a mock OptimizationProblem instance."""
    return MockOptimizationProblem()

@pytest.fixture
def mga_instance(mock_problem):
    """Provides a fresh MGAProblem instance for each test."""
    return mhmga.MGAProblem(problem=mock_problem, random_seed=42)

# Tests
def test_mga_init(mga_instance):
    """Tests the initialization of the MGAProblem class and attribute types."""
    # Assert that all attributes are initialized with the correct types
    # Assert that all attributes are initialized with the correct types
    assert isinstance(mga_instance.problem, MockOptimizationProblem), f"'problem' has bad type. Expected: MockOptimizationProblem, got: {type(mga_instance.problem)}"
    assert isinstance(mga_instance.rng, np.random.Generator), f"'rng' has bad type. Expected: np.random.Generator, got: {type(mga_instance.rng)}"
    assert is_boolean(mga_instance.stable_sort), f"'stable_sort' has bad type. Expected: bool, got: {type(mga_instance.stable_sort)}"
    assert mga_instance.logger is None, "logger should be None when no log_dir is provided"
    assert mga_instance.population is None, "population should be None at initialization"
    assert is_integer(mga_instance.current_iter), f"'current_iter' has bad type. Expected: int, got: {type(mga_instance.current_iter)}"
    assert isinstance(mga_instance.start_time, datetime), f"'start_time' has bad type. Expected: datetime, got: {type(mga_instance.start_time)}"
    assert is_boolean(mga_instance._is_populated), f"'_is_populated' has bad type. Expected: bool, got: {type(mga_instance._is_populated)}"
    assert is_integer(mga_instance.pop_size), f"'pop_size' has bad type. Expected: int, got: {type(mga_instance.pop_size)}"
    assert is_integer(mga_instance.elite_count), f"'elite_count' has bad type. Expected: int, got: {type(mga_instance.elite_count)}"
    assert is_integer(mga_instance.tourn_count), f"'tourn_count' has bad type. Expected: int got: {type(mga_instance.tourn_count)}"
    assert is_integer(mga_instance.tourn_size), f"'tourn_size' has bad type. Expected: int, got: {type(mga_instance.tourn_size)}"
    assert is_float(mga_instance.mutation_prob), f"'mutation_prob' has bad type. Expected: float, got: {type(mga_instance.mutation_prob)}"
    assert is_float(mga_instance.mutation_sigma), f"'mutation_sigma' has bad type. Expected: float, got: {type(mga_instance.mutation_sigma)}"
    assert is_float(mga_instance.crossover_prob), f"'crossover_prob' has bad type. Expected: float, got: {type(mga_instance.crossover_prob)}"
    assert mga_instance.niche_elitism is None, "niche_elitism should be None at initialization"
    assert is_float(mga_instance.noptimal_slack), f"'noptimal_slack' has bad type. Expected: float, got: {type(mga_instance.noptimal_slack)}"


def test_mga_init_with_logger(mock_problem):
    """Tests initialization with a logger directory and ensures cleanup."""
    # Ensure the directory does not exist before the test
    if os.path.exists("test_logs"):
        os.chmod("test_logs", 0o666)
        shutil.rmtree("test_logs", ignore_errors=True)
    try:
        mga_prob = mhmga.MGAProblem(problem=mock_problem, log_dir="test_logs/log", log_freq=1)
        assert isinstance(mga_prob.logger, Logger), f"'MGAProblem.logger' is not a Logger, Got: {type(mga_prob.logger)}"
        assert mga_prob.logger is not None, "Logger should be instantiated when log_dir is provided"
        assert os.path.exists("test_logs"), "Log directory should be created"
    finally:
        # This block will run whether the test fails or succeeds
        if os.path.exists("test_logs"):
            os.chmod("test_logs", 0o666)
            shutil.rmtree("test_logs")

def test_add_niches_calls_population_method(mga_instance):
    """Tests that MGAProblem.add_niches correctly calls Population.add_niches."""
    mga_instance.add_niches(5)
    assert mga_instance.num_niches == 5, "num_niches should be set before population is created"
    assert mga_instance.population is None, "population should not initiated until explicitly called"
    # Simulate population creation
    mga_instance.populate(pop_size=10, noptimal_slack=1.1, violation_factor=1.0)
    assert isinstance(mga_instance.population, Population), "population should be initiated after `mga.populate`"
    # Call add_niches again
    mga_instance.add_niches(3)
    # Verify that the population's add_niches method was called with the correct argument
    assert mga_instance.population.num_niches_called == 3, "'num_niches' not passed correctly to 'Population'"
    assert mga_instance.population.add_niches_calls == 1, "'mga.add_niches' did not call 'mga.Population.add_niches'"
    assert mga_instance.num_niches == 8, "Total niches should be updated after adding more"

def test_step_runs(mga_instance):
    """Tests that the main step function runs for a few iterations without errors."""
    mga_instance.add_niches(2)
    mga_instance.step(max_iter=5)
    assert mga_instance.current_iter == 5, "The algorithm should run for the specified number of iterations"

def test_step_input_validation(mga_instance):
    """Tests the comprehensive input validation for the step method."""
    mga_instance.add_niches(2)
    
    # Test max_iter
    with pytest.raises(TypeError, match="'max_iter' expected an int"):
        mga_instance.step(max_iter=5.5)
    with pytest.raises(ValueError, match="'max_iter' must be a strictly positive integer"):
        mga_instance.step(max_iter=0)

    # Test pop_size
    with pytest.raises(TypeError, match="'pop_size' expected an int"):
        mga_instance.step(max_iter=5, pop_size=100.5)
    with pytest.raises(ValueError, match="'pop_size' must be a strictly positive integer"):
        mga_instance.step(max_iter=5, pop_size=0)
        
    # Test elite_count
    with pytest.raises(TypeError, match="'elite_count' expected an int or float"):
        mga_instance.step(max_iter=5, elite_count="a")
    with pytest.raises(ValueError, match="float 'elite_count' should be in range"):
        mga_instance.step(max_iter=5, elite_count=1.1)
        
    # Test tourn_count
    with pytest.raises(TypeError, match="'tourn_count' expected an int or float"):
        mga_instance.step(max_iter=5, tourn_count=None)
    with pytest.raises(ValueError, match="float 'tourn_count' should be in range"):
        mga_instance.step(max_iter=5, tourn_count=-0.1)

    # Test tourn_size
    with pytest.raises(TypeError, match="'tourn_size' expected an integer"):
        mga_instance.step(max_iter=5, tourn_size=2.5)
    with pytest.raises(ValueError, match="'tourn_size' must be a strictly greater than 1"):
        mga_instance.step(max_iter=5, tourn_size=0)
        
    # Test niche_elitism
    with pytest.raises(ValueError, match="'niche_elitism' expected one of"):
        mga_instance.step(max_iter=5, niche_elitism="invalid_option")

    # Test multi-input constraints
    with pytest.raises(ValueError, match="'elite_count' \\+ 'tourn_count' should be weakly less than 'pop_size'"):
        mga_instance.step(max_iter=5, pop_size=100, elite_count=50, tourn_count=60)
    with pytest.raises(ValueError, match="'tourn_size' should be less than 'pop_size'"):
        mga_instance.step(max_iter=5, pop_size=10, tourn_size=11)

def test_populate(mga_instance):
    """Tests that the population is created and initialized correctly."""
    mga_instance.add_niches(3)
    assert not mga_instance._is_populated, "Should not be populated before calling populate"
    
    mga_instance.populate(pop_size=20, noptimal_slack=1.1, violation_factor=1.0)
    
    assert mga_instance._is_populated, "Should be populated after calling populate"
    assert mga_instance.population is not None, "Population object should be created"
    assert mga_instance.population.pop_size == 20, "Population size should be set correctly"

def test_get_results(mga_instance):
    """Tests that get_results returns the correct objects from the Population instance."""
    with pytest.raises(RuntimeError, match="Algorithm has not been run yet."):
        mga_instance.get_results()

    mga_instance.add_niches(2)
    mga_instance.step(max_iter=1)
    
    results = mga_instance.get_results()
    
    # Use 'is' to check for object identity, ensuring the correct attributes are returned
    assert results["optima"] is mga_instance.population.current_optima, "Results 'optima' should be the same object as population's optima"
    assert results["fitness"] is mga_instance.population.current_optima_fit, "Results 'fitness' should be the same object as population's fitness"
    assert results["objective"] is mga_instance.population.current_optima_obj, "Results 'objective' should be the same object as population's objective"
    assert results["noptimality"] is mga_instance.population.current_optima_nop, "Results 'noptimality' should be the same object as population's noptimality"

