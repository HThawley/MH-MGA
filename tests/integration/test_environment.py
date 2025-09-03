import pytest 
import numpy as np
import importlib
import sys

@pytest.fixture
def dummy():
    def func(arr):
        return arr.sum()
    yield func

@pytest.fixture
def bounds():
    yield (np.zeros(2), np.ones(2))

def test_precision_updating_good_32(dummy, bounds):
    for module in ("mga.commons.types", "mga.problem_definition", "mga.mga"):
        try: 
            importlib.reload(sys.modules[module])
        except KeyError:
            pass

    from mga.commons.types import DEFAULTS
    DEFAULTS.update_precision(32)
    from mga.problem_definition import OptimizationProblem
    from mga.mga import MGAProblem

    problem = OptimizationProblem(dummy, bounds)
    algorithm = MGAProblem(problem)
    algorithm.add_niches(2)
    algorithm.step(max_iter=2, pop_size=2)
    assert algorithm.problem.lower_bounds.dtype == np.float32, f"MGAProblem.problem.lower_bounds bad dtype: {algorithm.problem.lower_bounds.dtype}"
    assert algorithm.problem.upper_bounds.dtype == np.float32, f"MGAProblem.problem.upper_bounds bad dtype: {algorithm.problem.upper_bounds.dtype}"
    assert algorithm.population.points.dtype == np.float32, f"MGAProblem.population.points bad dtype: {algorithm.population.points.dtype}"
    assert algorithm.population.objective_values.dtype == np.float32, f"MGAProblem.population.objective_values bad dtype: {algorithm.population.objective_values.dtype}"
    assert algorithm.population.violations.dtype == np.float32, f"MGAProblem.population.violations bad dtype: {algorithm.population.violations.dtype}"
    assert algorithm.population.penalized_objectives.dtype == np.float32, f"MGAProblem.population.penalized_objectives bad dtype: {algorithm.population.penalized_objectives.dtype}"
    assert algorithm.population.fitnesses.dtype == np.float32, f"MGAProblem.population.fitnesses bad dtype: {algorithm.population.fitnesses.dtype}"
    assert algorithm.population.centroids.dtype == np.float32, f"MGAProblem.population.centroids bad dtype: {algorithm.population.centroids.dtype}"
    assert algorithm.population.niche_elites.dtype == np.float32, f"MGAProblem.population.niche_elites bad dtype: {algorithm.population.niche_elites.dtype}"
    assert algorithm.population.current_optima.dtype == np.float32, f"MGAProblem.population.current_optima bad dtype: {algorithm.population.current_optima.dtype}"
    assert algorithm.population.current_optima_obj.dtype == np.float32, f"MGAProblem.population.current_optima_obj bad dtype: {algorithm.population.current_optima_obj.dtype}"
    assert algorithm.population.current_optima_pob.dtype == np.float32, f"MGAProblem.population.current_optima_pob bad dtype: {algorithm.population.current_optima_pob.dtype}"
    assert algorithm.population.current_optima_fit.dtype == np.float32, f"MGAProblem.population.current_optima_fit bad dtype: {algorithm.population.current_optima_fit.dtype}"

    assert isinstance(algorithm.population.num_niches, np.int32), f"MGAProblem.population.num_niches bad dtype: {type(algorithm.population.num_niches)}"
    assert isinstance(algorithm.population.pop_size, np.int32), f"MGAProblem.population.pop_size bad dtype: {type(algorithm.population.pop_size)}"
    assert isinstance(algorithm.population.parent_size, np.int32), f"MGAProblem.population.parent_size bad dtype: {type(algorithm.population.parent_size)}"
    assert isinstance(algorithm.pop_size, np.int32), f"algorithm.pop_size bad dtype: {type(algorithm.pop_size)}"
    assert isinstance(algorithm.elite_count, np.int32), f"MGAProblem.elite_count bad dtype: {type(algorithm.elite_count)}"
    assert isinstance(algorithm.tourn_count, np.int32), f"MGAProblem.tourn_count bad dtype: {type(algorithm.tourn_count)}"
    assert isinstance(algorithm.tourn_size, np.int32), f"MGAProblem.tourn_size bad dtype: {type(algorithm.tourn_size)}"
    assert isinstance(algorithm.noptimal_slack, np.float32), f"MGAProblem.noptimal_slack bad dtype: {type(algorithm.noptimal_slack)}"
    if isinstance(algorithm.mutation_prob, np.ndarray):
        assert algorithm.mutation_prob.dtype == np.float32, f"MGAProblem.mutation_prob bad dtype: {algorithm.mutation_prob.dtype}"
    else: 
        assert isinstance(algorithm.mutation_prob, np.float32), f"MGAProblem.mutation_prob bad dtype: {type(algorithm.mutation_prob)}"
    if isinstance(algorithm.mutation_sigma, np.ndarray):
        assert algorithm.mutation_sigma.dtype == np.float32, f"MGAProblem.mutation_sigma bad dtype: {algorithm.mutation_sigma.dtype}"
    else: 
        assert isinstance(algorithm.mutation_sigma, np.float32), f"MGAProblem.mutation_sigma bad dtype: {type(algorithm.mutation_sigma)}"
    if isinstance(algorithm.crossover_prob, np.ndarray):
        assert algorithm.crossover_prob.dtype == np.float32, f"MGAProblem.crossover_prob bad dtype: {algorithm.crossover_prob.dtype}"
    else: 
        assert isinstance(algorithm.crossover_prob, np.float32), f"MGAProblem.crossover_prob bad dtype: {type(algorithm.crossover_prob)}"

def test_precision_updating_good_64(dummy, bounds):
    for module in ("mga.commons.types", "mga.problem_definition", "mga.mga"):
        try: 
            importlib.reload(sys.modules[module])
        except KeyError:
            pass
        
    from mga.commons.types import DEFAULTS
    DEFAULTS.update_precision(64)
    from mga.problem_definition import OptimizationProblem
    from mga.mga import MGAProblem

    problem = OptimizationProblem(dummy, bounds)
    algorithm = MGAProblem(problem)
    algorithm.add_niches(2)
    algorithm.step(max_iter=2, pop_size=2)
    assert algorithm.problem.lower_bounds.dtype == np.float64, f"MGAProblem.problem.lower_bounds bad dtype: {algorithm.problem.lower_bounds.dtype}"
    assert algorithm.problem.upper_bounds.dtype == np.float64, f"MGAProblem.problem.upper_bounds bad dtype: {algorithm.problem.upper_bounds.dtype}"
    assert algorithm.population.points.dtype == np.float64, f"MGAProblem.population.points bad dtype: {algorithm.population.points.dtype}"
    assert algorithm.population.objective_values.dtype == np.float64, f"MGAProblem.population.objective_values bad dtype: {algorithm.population.objective_values.dtype}"
    assert algorithm.population.violations.dtype == np.float64, f"MGAProblem.population.violations bad dtype: {algorithm.population.violations.dtype}"
    assert algorithm.population.penalized_objectives.dtype == np.float64, f"MGAProblem.population.penalized_objectives bad dtype: {algorithm.population.penalized_objectives.dtype}"
    assert algorithm.population.fitnesses.dtype == np.float64, f"MGAProblem.population.fitnesses bad dtype: {algorithm.population.fitnesses.dtype}"
    assert algorithm.population.centroids.dtype == np.float64, f"MGAProblem.population.centroids bad dtype: {algorithm.population.centroids.dtype}"
    assert algorithm.population.niche_elites.dtype == np.float64, f"MGAProblem.population.niche_elites bad dtype: {algorithm.population.niche_elites.dtype}"
    assert algorithm.population.current_optima.dtype == np.float64, f"MGAProblem.population.current_optima bad dtype: {algorithm.population.current_optima.dtype}"
    assert algorithm.population.current_optima_obj.dtype == np.float64, f"MGAProblem.population.current_optima_obj bad dtype: {algorithm.population.current_optima_obj.dtype}"
    assert algorithm.population.current_optima_pob.dtype == np.float64, f"MGAProblem.population.current_optima_pob bad dtype: {algorithm.population.current_optima_pob.dtype}"
    assert algorithm.population.current_optima_fit.dtype == np.float64, f"MGAProblem.population.current_optima_fit bad dtype: {algorithm.population.current_optima_fit.dtype}"

    assert isinstance(algorithm.population.num_niches, np.int64), f"MGAProblem.population.num_niches bad dtype: {type(algorithm.population.num_niches)}"
    assert isinstance(algorithm.population.pop_size, np.int64), f"MGAProblem.population.pop_size bad dtype: {type(algorithm.population.pop_size)}"
    assert isinstance(algorithm.population.parent_size, np.int64), f"MGAProblem.population.parent_size bad dtype: {type(algorithm.population.parent_size)}"
    assert isinstance(algorithm.pop_size, np.int64), f"MGAProblem.pop_size bad dtype: {type(algorithm.pop_size)}"
    assert isinstance(algorithm.elite_count, np.int64), f"MGAProblem.elite_count bad dtype: {type(algorithm.elite_count)}"
    assert isinstance(algorithm.tourn_count, np.int64), f"MGAProblem.tourn_count bad dtype: {type(algorithm.tourn_count)}"
    assert isinstance(algorithm.tourn_size, np.int64), f"MGAProblem.tourn_size bad dtype: {type(algorithm.tourn_size)}"
    assert isinstance(algorithm.noptimal_slack, np.float64), f"MGAProblem.noptimal_slack bad dtype: {type(algorithm.noptimal_slack)}"
    if isinstance(algorithm.mutation_prob, np.ndarray):
        assert algorithm.mutation_prob.dtype == np.float64, f"MGAProblem.mutation_prob bad dtype: {algorithm.mutation_prob.dtype}"
    else: 
        assert isinstance(algorithm.mutation_prob, np.float64), f"MGAProblem.mutation_prob bad dtype: {type(algorithm.mutation_prob)}"
    if isinstance(algorithm.mutation_sigma, np.ndarray):
        assert algorithm.mutation_sigma.dtype == np.float64, f"MGAProblem.mutation_sigma bad dtype: {algorithm.mutation_sigma.dtype}"
    else: 
        assert isinstance(algorithm.mutation_sigma, np.float64), f"MGAProblem.mutation_sigma bad dtype: {type(algorithm.mutation_sigma)}"
    if isinstance(algorithm.crossover_prob, np.ndarray):
        assert algorithm.crossover_prob.dtype == np.float64, f"MGAProblem.crossover_prob bad dtype: {algorithm.crossover_prob.dtype}"
    else: 
        assert isinstance(algorithm.crossover_prob, np.float64), f"MGAProblem.crossover_prob bad dtype: {type(algorithm.crossover_prob)}"