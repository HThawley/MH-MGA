import numpy as np
from numba import njit
import warnings 

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
from mga.problem_definition import MultiObjectiveProblem
from mga.operators import selection, crossover, mutation
from mga.metrics import fitness as fit_metrics


class Pareto:
    def __init__(
            self, 
            problem: MultiObjectiveProblem, 
            pop_size: int, 
            rng: np.random._generator.Generator,
            stable_sort: bool,
        ):
        self.problem = problem
        self.pop_size = INT(pop_size)
        self.pareto_size = INT(0)
        self.rng = rng
        self.stable_sort = stable_sort
        
        # Population data arrays
        self.points = np.empty((pop_size, problem.ndim), dtype=FLOAT)
        self.objective_values = np.empty((pop_size, self.problem.n_objs), dtype=FLOAT)
        self.is_feasible = np.empty((pop_size, self.problem.n_objs), dtype=np.bool_)

        # Overall best found
        self.pareto = np.empty((0, problem.ndim), dtype=FLOAT)
        self.pareto_objs = np.empty((0, problem.n_objs), dtype=FLOAT)

        # Auxiliary functions
        self.cx_func = crossover._cx_two_point if problem.ndim > 2 else crossover._cx_one_point
        self.mut_func = (mutation.mutate_gaussian_niche_float 
                         if self.problem.integrality.sum() == 0 else 
                         mutation.mutate_gaussian_niche_mixed)

    def populate(self, x0=None):
        if x0 is None:
            self._populate_randomly()
            self._apply_integrality()
            self._apply_bounds()
            self._evaluate()
        else: 
            _clone(self.points, np.atleast_2d(x0))
            self._apply_integrality()
            self._apply_bounds()
            x0_obj, x0_fea = self.problem.evaluate(np.atleast_2d(x0))
            self.objective_values[:] = x0_obj[0, :]
            self.is_feasible[:] = x0_fea[0, :]
        

    def resize(self, pop_size: int):
        new_points = np.empty((pop_size, self.ndim), FLOAT)
        new_objectives = np.empty((pop_size, self.problem.n_objs), FLOAT)
        new_feasible = np.empty((pop_size, self.problem.n_objs), np.bool_)

        if pop_size < self.pop_size:
            raise ValueError("Reducing 'pop_size' may lose pareto efficient points")
        else: 
            _clone(new_points, self.points)
            _clone(new_objectives, self.objective_values)
            _clone(new_feasible, self.is_feasible)
            self.points = new_points
            self.objective_values = new_objectives
            self.is_feasible = new_feasible
            
        self.pop_size = pop_size

    def evolve(
            self, 
            npareto: int,
            mutation_prob: float,
            mutation_sigma: float, 
            crossover_prob: float, 
            ):
        self.npareto = npareto
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.crossover_prob = crossover_prob

        self._select_pareto()
        self._generate_offspring()
        self._evaluate()

    def _select_pareto(self):
        self.pareto, self.pareto_objs = _select_pareto(
            self.points, self.objective_values, self.problem.maximize, self.is_feasible
        )

    def _generate_offspring(self):
        _clone(self.points, self.pareto)

        crossover.crossover_niche(self.points, self.mutation_prob, self.cx_func, self.rng, start_idx=0)
        self.mut_func(self.points, self.mutation_sigma, self.mutation_prob, 
                      self.rng, self.problem.integrality, self.problem.boolean_mask, startidx=0)
        self._apply_bounds()
        _overwrite(self.points, self.pareto)
        

    def _evaluate(self):
        """
        Evaluates objectives
        """
        current_pareto_size = self.pareto.shape[0]
        # Evaluate objectives and apply penalties
        self.objective_values[current_pareto_size:], self.is_feasible[current_pareto_size:] = self.problem.evaluate(self.points[current_pareto_size:])
        self.objective_values[:current_pareto_size] = self.pareto_objs[:]
        self.is_feasible[:current_pareto_size] = True
        


    def _apply_bounds(self):
        """
        Clips the population points to stay within the defined bounds.
        """
        _apply_bounds(
            self.points,
            self.problem.lower_bounds,
            self.problem.upper_bounds,
        )

    def _apply_integrality(self):
        """
        Rounds points for variables that are defined as integers.
        """
        for k in range(self.problem.ndim):
            if self.problem.integrality[k]:
                np.round(self.points[:, k], out=self.points[:, k])


    def _populate_randomly(self):
        """Helper to populate a slice of niches with random points."""
        _populate_randomly(
            self.points, 
            self.problem.lower_bounds,
            self.problem.upper_bounds,
            self.rng,
        )

#%%

@njit
def _populate_randomly(points, lb, ub, rng):
    for j in range(points.shape[0]):
        for k in range(points.shape[1]):
            points[j, k] = rng.uniform(lb[k], ub[k])   

@njit
def _apply_bounds(points, lb, ub):
    for j in range(points.shape[0]):
        for k in range(points.shape[1]):
            points[j, k] = min(ub[k], max(lb[k], points[j, k]))

@njit
def _overwrite(target, elites):
    for j in range(elites.shape[0]):
        for k in range(elites.shape[1]):
            target[j, k] = elites[j, k]

@njit
def _clone(target, start_pop):
    nindividuals = start_pop.shape[0]
    for j in range(target.shape[0]):
        jn = j%nindividuals
        for k in range(target.shape[1]):
            target[j, k] = start_pop[jn, k]

@njit
def _select_pareto(points, objective, maximize, is_feasible):
    popsize, n_objs = objective.shape

    feasible_mask = np.ones(popsize, dtype=np.bool_)
    for i in range(popsize):
        for j in range(n_objs):
            if not is_feasible[i, j]:
                feasible_mask[i] = False
                break

    feasible_indices = np.where(feasible_mask)[0]

    if feasible_indices.size == 0:
        return (np.empty((0, points.shape[1]), points.dtype),
                np.empty((0, n_objs), objective.dtype))

    nfeas = feasible_indices.shape[0]
    processed_obj = np.empty((nfeas, n_objs), dtype=objective.dtype)
    for idx_j, j in enumerate(feasible_indices):
        for n in range(n_objs):
            if maximize[n]:
                processed_obj[idx_j, n] = -objective[j, n]
            else:
                processed_obj[idx_j, n] = objective[j, n]

    pareto_local_indices = np.empty(nfeas, dtype=np.intp)
    pareto_count = 0

    for j in range(nfeas):
        candidate_obj = processed_obj[j]
        dominated = False

        dominated_mask = np.zeros(pareto_count, dtype=np.bool_)

        for k in range(pareto_count):
            current_idx = pareto_local_indices[k]
            current_obj = processed_obj[current_idx]

            candidate_better = False
            current_better = False
            for n in range(n_objs):
                if candidate_obj[n] < current_obj[n]:
                    candidate_better = True
                elif candidate_obj[n] > current_obj[n]:
                    current_better = True

            if candidate_better and not current_better:
                dominated_mask[k] = True
            elif current_better and not candidate_better:
                dominated = True
                break

        if dominated:
            continue

        write_idx = 0
        for k in range(pareto_count):
            if not dominated_mask[k]:
                pareto_local_indices[write_idx] = pareto_local_indices[k]
                write_idx += 1

        pareto_local_indices[write_idx] = j
        pareto_count = write_idx + 1

    final_indices = feasible_indices[pareto_local_indices[:pareto_count]]
    pareto_points = points[final_indices]
    pareto_objs = objective[final_indices]

    return pareto_points, pareto_objs

