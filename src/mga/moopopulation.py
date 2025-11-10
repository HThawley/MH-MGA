import numpy as np
from numba import njit

from mga.commons.types import DEFAULTS

INT, FLOAT = DEFAULTS
from mga.problem_definition import MultiObjectiveProblem  # noqa: E402
from mga.operators import selection, crossover, mutation  # noqa: E402


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
        self.parents = np.empty((0, problem.ndim), dtype=FLOAT)
        # Overall best found
        self.pareto = np.empty((0, problem.ndim), dtype=FLOAT)
        self.pareto_objs = np.empty((0, problem.n_objs), dtype=FLOAT)

        # NSGA-II
        self.ranks = np.empty(pop_size, dtype=INT)
        self.crowding_distances = np.empty(pop_size, dtype=FLOAT)
        self.fitness_scores = np.empty(pop_size, dtype=FLOAT)

        # Auxiliary functions
        self.cx_func = crossover._cx_two_point if problem.ndim > 2 else crossover._cx_one_point
        self.mut_func = (
            mutation.mutate_gaussian_niche_float
            if self.problem.integrality.sum() == 0
            else mutation.mutate_gaussian_niche_mixed
        )

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
        self._update_ranks_and_crowding()

    def resize(self, pop_size: int = None, parent_size: int = None):
        if pop_size is not None:
            self._resize_pop_size(pop_size)
        if parent_size is not None:
            self._resize_parent_size(parent_size)

    def _resize_parent_size(self, parent_size):
        parents = np.empty((parent_size, self.problem.ndim), FLOAT)
        if self.parents.shape[0] > 0:
            _clone(parents, self.parents)
        self.parents = parents

    def _resize_pop_size(self, pop_size):
        points = np.empty((pop_size, self.problem.ndim), FLOAT)
        objectives = np.empty((pop_size, self.problem.n_objs), FLOAT)
        feasible = np.empty((pop_size, self.problem.n_objs), np.bool_)

        if pop_size < self.pop_size:
            # When reducing, keep best solutions based on rank and crowding
            best_indices = self._select_best_solutions(pop_size)
            points[:] = self.points[best_indices]
            objectives[:] = self.objective_values[best_indices]
            feasible[:] = self.is_feasible[best_indices]
        else:
            _clone(points, self.points)
            _clone(objectives, self.objective_values)
            _clone(feasible, self.is_feasible)
            feasible[: self.pareto.shape[0], :] = True

        self.points = points
        self.objective_values = objectives
        self.is_feasible = feasible
        self.ranks = np.empty(pop_size, dtype=INT)
        self.crowding_distances = np.empty(pop_size, dtype=FLOAT)
        self.fitness_scores = np.empty(pop_size, dtype=FLOAT)

        self.pop_size = pop_size

    def evolve(
        self,
        pareto_size: int,
        elite_count: int,
        tourn_count: int,
        tourn_size: int,
        mutation_prob: float,
        mutation_sigma: float,
        crossover_prob: float,
    ):
        self.pareto_size = pareto_size
        self.elite_count = elite_count
        self.tourn_count = tourn_count
        self.tourn_size = tourn_size

        if self.parents.shape[0] != self.elite_count + self.tourn_count:
            self.resize(parent_size=self.elite_count + self.tourn_count)

        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.crossover_prob = crossover_prob

        self._update_ranks_and_crowding()
        self._select_pareto()

        self._select_parents()
        self._generate_offspring()
        self._evaluate()

    def _select_parents(self):
        selection.selection(
            self.parents,
            self.points,
            self.ranks,
            False,
            self.elite_count,
            self.tourn_count,
            self.tourn_size,
            self.rng,
            self.stable_sort,
        )

    def _select_pareto(self):
        self.pareto, self.pareto_objs = _select_pareto(
            self.points, self.objective_values, self.problem.maximize, self.is_feasible
        )

    def _generate_offspring(self):
        _clone(self.points, self.parents)

        crossover.crossover_niche(self.points, self.mutation_prob, self.cx_func, self.rng, start_idx=0)
        self.mut_func(
            self.points,
            self.mutation_sigma,
            self.mutation_prob,
            self.rng,
            self.problem.integrality,
            self.problem.boolean_mask,
            startidx=0,
        )
        self._apply_bounds()
        self._apply_integrality()

    def _evaluate(self):
        """
        Evaluates objectives
        """
        # Evaluate objectives and apply penalties
        self.objective_values[:], self.is_feasible[:] = self.problem.evaluate(self.points)

    def _update_ranks_and_crowding(self):
        """
        Update ranks using non-dominated sorting and calculate crowding distances.
        """
        # Get feasible solutions
        feasible_mask = np.all(self.is_feasible, axis=1)
        n_feasible = np.sum(feasible_mask)

        if n_feasible == 0:
            self.ranks[:] = 0
            self.crowding_distances[:] = 0
            return

        # Non-dominated sorting
        self.ranks[:] = self.ranks.shape[0] + 1  # Initialize with worst rank
        remaining_indices = np.where(feasible_mask)[0]
        current_rank = 0

        while len(remaining_indices) > 0:
            # Find non-dominated solutions in remaining set
            non_dominated_mask = _find_non_dominated(self.objective_values[remaining_indices], self.problem.maximize)

            non_dominated_indices = remaining_indices[non_dominated_mask]
            self.ranks[non_dominated_indices] = current_rank

            # Calculate crowding distance for this front
            self._calculate_crowding_distance(non_dominated_indices)

            # Remove non-dominated from remaining
            remaining_indices = remaining_indices[~non_dominated_mask]
            current_rank += 1

        # Assign worst rank to infeasible solutions
        self.ranks[~feasible_mask] = current_rank
        self.crowding_distances[~feasible_mask] = 0

    def _calculate_crowding_distance(self, indices: np.ndarray):
        """
        Calculate crowding distance for a set of solutions.
        """
        if len(indices) <= 2:
            self.crowding_distances[indices] = np.inf
            return

        n_solutions = len(indices)
        distances = np.zeros(n_solutions)

        for obj_idx in range(self.problem.n_objs):
            # Sort by objective
            obj_values = self.objective_values[indices, obj_idx]
            sorted_idx = np.argsort(obj_values)

            # Boundary points get infinite distance
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf

            # Calculate distances for interior points
            obj_range = obj_values[sorted_idx[-1]] - obj_values[sorted_idx[0]]
            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    distances[sorted_idx[i]] += (
                        obj_values[sorted_idx[i + 1]] - obj_values[sorted_idx[i - 1]]
                    ) / obj_range

        self.crowding_distances[indices] = distances

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


# %%


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
        jn = j % nindividuals
        for k in range(target.shape[1]):
            target[j, k] = start_pop[jn, k]


@njit
def _find_non_dominated(objectives, maximize):  # noqa: C901
    """
    Find non-dominated solutions in objective space.
    """
    n_solutions = objectives.shape[0]
    n_objs = objectives.shape[1]
    is_non_dominated = np.ones(n_solutions, dtype=np.bool_)

    for i in range(n_solutions):
        if not is_non_dominated[i]:
            continue

        for j in range(n_solutions):
            if i == j or not is_non_dominated[j]:
                continue

            # Check if j dominates i
            j_dominates_i = True
            j_better_in_at_least_one = False

            for obj_idx in range(n_objs):
                if maximize[obj_idx]:
                    if objectives[j, obj_idx] < objectives[i, obj_idx]:
                        j_dominates_i = False
                        break
                    elif objectives[j, obj_idx] > objectives[i, obj_idx]:
                        j_better_in_at_least_one = True
                else:
                    if objectives[j, obj_idx] > objectives[i, obj_idx]:
                        j_dominates_i = False
                        break
                    elif objectives[j, obj_idx] < objectives[i, obj_idx]:
                        j_better_in_at_least_one = True

            if j_dominates_i and j_better_in_at_least_one:
                is_non_dominated[i] = False
                break

    return is_non_dominated


@njit
def _select_pareto(points, objective, maximize, is_feasible):  # noqa: C901
    popsize, n_objs = objective.shape

    feasible_mask = np.ones(popsize, dtype=np.bool_)
    for i in range(popsize):
        for j in range(n_objs):
            if not is_feasible[i, j]:
                feasible_mask[i] = False
                break

    feasible_indices = np.where(feasible_mask)[0]

    if feasible_indices.size == 0:
        return (np.empty((0, points.shape[1]), points.dtype), np.empty((0, n_objs), objective.dtype))

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

            if (not candidate_better) and (not current_better):
                dominated = True
                break
            elif candidate_better and not current_better:
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
