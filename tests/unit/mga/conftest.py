import pytest
import numpy as np

from mga.commons.types import DEFAULTS

INT, FLOAT = DEFAULTS
import mga.population as pp  # noqa: E402
from mga.mhmga import MGAProblem  # noqa: E402
from mga.problem_definition import OptimizationProblem  # noqa: E402


class MockOptimizationProblem(OptimizationProblem):
    def __init__(self, ndim=3, maximize=True):
        self.ndim = ndim
        self.lower_bounds = np.zeros(ndim, dtype=FLOAT)
        self.upper_bounds = np.ones(ndim, dtype=FLOAT)
        self.integrality = np.zeros(ndim, dtype=bool)
        self.booleanality = np.zeros(ndim, dtype=bool)
        self.known_optimum_point = np.full(ndim, 0.9, dtype=FLOAT)
        self.known_optimum_value = -0.43 if maximize else 2.43
        self.maximize = maximize

    def re__init__(self, **kwargs):
        if "ndim" in kwargs.keys():
            self.ndim = kwargs["ndim"]
            self.lower_bounds = np.zeros(self.ndim, dtype=FLOAT)
            self.upper_bounds = np.ones(self.ndim, dtype=FLOAT)
            self.integrality = np.zeros(self.ndim, dtype=bool)
            self.booleanality = np.zeros(self.ndim, dtype=bool)
            self.known_optimum_point = np.full(self.ndim, 0.9, dtype=FLOAT)
        if "maximize" in kwargs.keys():
            self.maximize = kwargs["maximize"]
            self.known_optimum_value = -0.43 if self.maximize else 2.43

    def evaluate(self, points):
        # Simple evaluation: sum of squares, negated for maximization
        objectives = 2 - np.sum(points**2, axis=1) if self.maximize else np.sum(points**2, axis=1)
        violations = np.zeros(points.shape[0])
        return objectives, violations

    def evaluate_population(self, population):
        for i in range(population.num_niches):
            population.objective_values[i], population.violations[i] = self.evaluate(population.points[i])
        population.penalized_objectives[:] = (
            population.objective_values + population.violations * population.violation_factor
        )


# fixtures
@pytest.fixture
def rng():
    yield np.random.default_rng()


# Fixtures
@pytest.fixture
def mock_problem():
    """Provides a mock OptimizationProblem instance."""
    yield MockOptimizationProblem(ndim=3)


@pytest.fixture
def mga_instance(mock_problem):
    """Provides a fresh MGAProblem instance for each test."""
    yield MGAProblem(problem=mock_problem, random_seed=42)


@pytest.fixture
def population_instance(mock_problem, rng):
    """Provides a Population instance for testing."""
    population = pp.Population(
        num_niches=2,
        pop_size=5,
        ndim=mock_problem.ndim,
        stable_sort=False,
        rng=rng,
    )
    pp.load_problem_to_population(population, mock_problem)
    yield population
