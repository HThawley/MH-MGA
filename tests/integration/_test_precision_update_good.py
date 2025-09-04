import numpy as np 
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-p', type=int, required=True, help='precision for test')
args = parser.parse_args()

def test_precision_update_good(precision):
    #SETUP
    def dummy(arr):
        return arr.sum()
    bounds=(np.zeros(2), np.ones(2))
    FLOAT = getattr(np, f"float{precision}")
    INT = getattr(np, f"int{precision}")

    #BODY
    from mga.commons.types import DEFAULTS
    DEFAULTS.update_precision(precision)
    from mga.problem_definition import OptimizationProblem
    from mga.mga import MGAProblem
    
    problem = OptimizationProblem(dummy, bounds)
    algorithm = MGAProblem(problem)
    algorithm.add_niches(2)
    algorithm.step(max_iter=2, pop_size=2)
    try: 
        assert algorithm.problem.lower_bounds.dtype == FLOAT, f"MGAProblem.problem.lower_bounds bad dtype: {algorithm.problem.lower_bounds.dtype}, expected {FLOAT}"
        assert algorithm.problem.upper_bounds.dtype == FLOAT, f"MGAProblem.problem.upper_bounds bad dtype: {algorithm.problem.upper_bounds.dtype}, expected {FLOAT}"
        assert algorithm.population.points.dtype == FLOAT, f"MGAProblem.population.points bad dtype: {algorithm.population.points.dtype}, expected {FLOAT}"
        assert algorithm.population.objective_values.dtype == FLOAT, f"MGAProblem.population.objective_values bad dtype: {algorithm.population.objective_values.dtype}, expected {FLOAT}"
        assert algorithm.population.violations.dtype == FLOAT, f"MGAProblem.population.violations bad dtype: {algorithm.population.violations.dtype}, expected {FLOAT}"
        assert algorithm.population.penalized_objectives.dtype == FLOAT, f"MGAProblem.population.penalized_objectives bad dtype: {algorithm.population.penalized_objectives.dtype}, expected {FLOAT}"
        assert algorithm.population.fitnesses.dtype == FLOAT, f"MGAProblem.population.fitnesses bad dtype: {algorithm.population.fitnesses.dtype}, expected {FLOAT}"
        assert algorithm.population.centroids.dtype == FLOAT, f"MGAProblem.population.centroids bad dtype: {algorithm.population.centroids.dtype}, expected {FLOAT}"
        assert algorithm.population.niche_elites.dtype == FLOAT, f"MGAProblem.population.niche_elites bad dtype: {algorithm.population.niche_elites.dtype}, expected {FLOAT}"
        assert algorithm.population.current_optima.dtype == FLOAT, f"MGAProblem.population.current_optima bad dtype: {algorithm.population.current_optima.dtype}, expected {FLOAT}"
        assert algorithm.population.current_optima_obj.dtype == FLOAT, f"MGAProblem.population.current_optima_obj bad dtype: {algorithm.population.current_optima_obj.dtype}, expected {FLOAT}"
        assert algorithm.population.current_optima_pob.dtype == FLOAT, f"MGAProblem.population.current_optima_pob bad dtype: {algorithm.population.current_optima_pob.dtype}, expected {FLOAT}"
        assert algorithm.population.current_optima_fit.dtype == FLOAT, f"MGAProblem.population.current_optima_fit bad dtype: {algorithm.population.current_optima_fit.dtype}, expected {FLOAT}"

        assert isinstance(algorithm.population.num_niches, INT), f"MGAProblem.population.num_niches bad dtype: {type(algorithm.population.num_niches)}, expected {INT}"
        assert isinstance(algorithm.population.pop_size, INT), f"MGAProblem.population.pop_size bad dtype: {type(algorithm.population.pop_size)}, expected {INT}"
        assert isinstance(algorithm.population.parent_size, INT), f"MGAProblem.population.parent_size bad dtype: {type(algorithm.population.parent_size)}, expected {INT}"
        assert isinstance(algorithm.pop_size, INT), f"algorithm.pop_size bad dtype: {type(algorithm.pop_size)}, expected {INT}"
        assert isinstance(algorithm.elite_count, INT), f"MGAProblem.elite_count bad dtype: {type(algorithm.elite_count)}, expected {INT}"
        assert isinstance(algorithm.tourn_count, INT), f"MGAProblem.tourn_count bad dtype: {type(algorithm.tourn_count)}, expected {INT}"
        assert isinstance(algorithm.tourn_size, INT), f"MGAProblem.tourn_size bad dtype: {type(algorithm.tourn_size)}, expected {INT}"
        assert isinstance(algorithm.noptimal_slack, FLOAT), f"MGAProblem.noptimal_slack bad dtype: {type(algorithm.noptimal_slack)}, expected {FLOAT}"
        if isinstance(algorithm.mutation_prob, np.ndarray):
            assert algorithm.mutation_prob.dtype == FLOAT, f"MGAProblem.mutation_prob bad dtype: {algorithm.mutation_prob.dtype}, expected {FLOAT}"
        else: 
            assert isinstance(algorithm.mutation_prob, FLOAT), f"MGAProblem.mutation_prob bad dtype: {type(algorithm.mutation_prob)}, expected {FLOAT}"
        if isinstance(algorithm.mutation_sigma, np.ndarray):
            assert algorithm.mutation_sigma.dtype == FLOAT, f"MGAProblem.mutation_sigma bad dtype: {algorithm.mutation_sigma.dtype}, expected {FLOAT}"
        else: 
            assert isinstance(algorithm.mutation_sigma, FLOAT), f"MGAProblem.mutation_sigma bad dtype: {type(algorithm.mutation_sigma)}, expected {FLOAT}"
        if isinstance(algorithm.crossover_prob, np.ndarray):
            assert algorithm.crossover_prob.dtype == FLOAT, f"MGAProblem.crossover_prob bad dtype: {algorithm.crossover_prob.dtype}, expected {FLOAT}"
        else: 
            assert isinstance(algorithm.crossover_prob, FLOAT), f"MGAProblem.crossover_prob bad dtype: {type(algorithm.crossover_prob)}, expected {FLOAT}"

    except AssertionError as e: 
        print(f"Assertion Error: {e}", file=sys.stderr) 
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    test_precision_update_good(args.p)