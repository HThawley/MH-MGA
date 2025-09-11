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
        self.paretos_objs = np.empty((0, problem.n_objs), dtype=FLOAT)

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

    def _evaluate(self):
        """
        Evaluates objectives
        """
        # Evaluate objectives and apply penalties
        self.objective_values[:], self.is_feasible[:] = self.problem.evaluate(self.points)
        
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
    """
    Select pareto efficeint solutions as parents
    """
    popsize, n_objs = objective.shape
    # Correctly handle minimization/maximization

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
    
    for j in range(nfeas):
        for n in range(n_objs):
            if maximize[n]:
                processed_obj[j, n] = -objective[feasible_indices[j], n]
            else:
                processed_obj[j, n] = objective[feasible_indices[j], n]

    pareto_indices_local = np.zeros(nfeas, dtype=np.int64)
    pareto_count = 0

    # Main Loop: Iterate through each candidate solution
    for j in range(nfeas):
        candidate_obj = processed_obj[j]
        is_candidate_dominated = False
        # Track which of the *current* Pareto solutions are dominated by the new candidate. 
        dominated_in_current_front_mask = np.zeros(pareto_count, dtype=np.bool_)
        #  Inner Loop: Compare candidate against the current Pareto front 
        for k in range(pareto_count):
            current_pareto_local_idx = pareto_indices_local[k]
            current_pareto_obj = processed_obj[current_pareto_local_idx]
            
            candidate_is_better = False
            pareto_is_better = False
            for n in range(n_objs):
                if candidate_obj[n] < current_pareto_obj[n]:
                    candidate_is_better = True
                elif current_pareto_obj[n] < candidate_obj[n]:
                    pareto_is_better = True
            
            if not candidate_is_better and not pareto_is_better: # They are equal
                is_candidate_dominated = True
                break
            elif pareto_is_better and not candidate_is_better: # Pareto point dominates candidate
                is_candidate_dominated = True
                break
            elif candidate_is_better and not pareto_is_better: # Candidate dominates Pareto point
                dominated_in_current_front_mask[k] = True
            
                # Update Pareto Front
        if is_candidate_dominated:
            continue
        # The candidate is not dominated. We now rebuild the Pareto set by:
        # 1. Keeping the old members that were NOT dominated by the candidate.
        # 2. Adding the new candidate.
        # This copies over the surviving indices to the front of the array.
        write_idx = 0
        for k in range(pareto_count):
            if not dominated_in_current_front_mask[k]:
                pareto_indices_local[write_idx] = pareto_indices_local[k]
                write_idx += 1
        # Add the new candidate's index to the end of the filtered list.
        pareto_indices_local[write_idx] = j
        # Update the count of solutions on the front.
        pareto_count = write_idx + 1
    final_local_indices = pareto_indices_local[:pareto_count]
    
    pareto = np.empty((pareto_count, points.shape[1]), points.dtype)
    pareto_objs = np.empty((pareto_count, n_objs), objective.dtype)

    final_indices = feasible_indices[final_local_indices]
    pareto[:] = points[final_indices, :]
    pareto_objs[:] = objective[final_indices, :]
    
    return pareto, pareto_objs