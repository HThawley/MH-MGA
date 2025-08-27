import numpy as np
from numba import njit
from scipy.spatial import Delaunay

from mga.problem_definition import OptimizationProblem
from mga.operators import selection, crossover, mutation
from mga.metrics import fitness as fit_metrics

### 
# Can this be made a jitclass??
###

class Population:
    """
    Manages the state and evolution of the entire population across all niches.
    """
    def __init__(
            self, 
            problem: OptimizationProblem, 
            num_niches: int, 
            pop_size: int, 
            rng: np.random._generator.Generator,
        ):
        self.problem = problem
        self.num_niches = num_niches
        self.pop_size = pop_size
        self.parent_size = 0
        self.rng = rng
        
        # Population data arrays
        self.points = np.empty((num_niches, pop_size, problem.ndim), dtype=np.float64)
        self.objective_values = np.empty((num_niches, pop_size), dtype=np.float64)
        self.violations = np.zeros((num_niches, pop_size), dtype=np.float64)
        self.penalized_objectives = np.empty((num_niches, pop_size), dtype=np.float64)
        self.fitnesses = np.empty((num_niches, pop_size), dtype=np.float64)
        self.is_noptimal = np.empty((num_niches, pop_size), dtype=np.bool_)
        self.centroids = np.empty((num_niches, problem.ndim), dtype=np.float64)
        self.niche_elites = np.empty((num_niches - 1, 1, problem.ndim), dtype=np.float64)
        
        # Overall best found
        self.global_best_point = np.copy(problem.known_optimum_point)
        self.global_best_value = problem.known_optimum_value
        self.noptimal_threshold = self.global_best_value

        self.cx_func = crossover._cx_two_point if problem.ndim > 2 else crossover._cx_one_point

    def initialize(self):
        """
        Initializes the population points with a uniform distribution
        """
        self._initialize_randomly(0, self.num_niches)
        self._apply_integrality()
        self._apply_bounds()

    def _initialize_randomly(self, start_idx, end_idx):
        """Helper to initialize a slice of niches with random points."""
        _initialize_randomly(
            self.points, 
            start_idx, 
            end_idx, 
            self.problem.lower_bounds,
            self.problem.upper_bounds,
            self.rng,
        )

    def add_niches(self, num_new_niches: int, heuristic: bool = False):
        """
        Adds niches to the population. 
        Heuristic uses delaunay triangulation to place the new niches inside 
        the convex hull of existing points. (Not tractable above ~8 dim)
        """
        if num_new_niches == 0:
            return
        if num_new_niches < 0:
            raise ValueError("Number of new niches must be positive")

        num_old_niches = self.num_niches
        num_total_niches = num_old_niches + num_new_niches

        self.points = _add_niche_to_array(self.points, num_total_niches)
        self.objective_values = _add_niche_to_array(self.objective_values, num_total_niches)
        self.violations = _add_niche_to_array(self.violations, num_total_niches)
        self.penalized_objectives = _add_niche_to_array(self.penalized_objectives, num_total_niches)
        self.fitnesses = _add_niche_to_array(self.fitnesses, num_total_niches)
        self.is_noptimal = _add_niche_to_array(self.is_noptimal, num_total_niches)
        self.centroids = _add_niche_to_array(self.centroids, num_total_niches)

        if heuristic and num_old_niches >= self.problem.ndim + 1:
            try:
                delaunay = Delaunay(self.centroids[:num_old_niches])
                candidates = []
                for simplex_indices in delaunay.simplices:
                    simplex_pts = self.centroids[simplex_indices]
                    A = 2 * (simplex_pts[1:] - simplex_pts[0])
                    b = np.sum(simplex_pts[1:]**2 - simplex_pts[0]**2, axis=1)
                    if np.linalg.det(A) != 0:
                        center = np.linalg.solve(A, b)
                        candidates.append(center)
                
                for i in range(num_new_niches):
                    niche_idx = num_old_niches + i
                    if i < len(candidates):
                         self.points[niche_idx, :, :] = candidates[i] + self.rng.normal(scale=0.01, size=self.points[niche_idx].shape)
                    else:
                        self._initialize_randomly(niche_idx, niche_idx + 1)
            except Exception:
                self._initialize_randomly(num_old_niches, num_total_niches)
        else:
             self._initialize_randomly(num_old_niches, num_total_niches)    

        self._initialize_randomly(num_old_niches, num_total_niches)
        
        self.num_niches = num_total_niches
        self._apply_integrality()
        self._apply_bounds()

    def resize(self, new_pop_size: int, new_parent_size:int, stable_sort: bool):
        """
        Resizes the population of each niche to a new size.
        """
        if new_pop_size != self.pop_size:
            # Create new arrays with the target size
            new_points = np.empty((self.num_niches, new_pop_size, self.problem.ndim))
            
            if new_pop_size > self.pop_size:
                clone(new_points, self.points)

            else: # new_pop_size < self.pop_size
                # Decrease size: select the best individuals to keep
                # Niche 0 (optimization) is selected based on objective value
                new_points[0] = selection.selection(
                    self.points[0], self.penalized_objectives[0], self.problem.maximize, 
                    self.elite_count, self.tourn_count, self.tourn_size, self.problem.rng, 
                    stable_sort)

                # Other niches (diversity) are selected based on fitness
                for i in range(1, self.num_niches):
                    new_points[i] = selection.selection_with_fallback(
                        self.points[i], self.fitnesses[i], self.is_noptimal[i],
                        self.penalized_objectives[i], self.problem.maximize, self.elite_count, 
                        self.tourn_count, self.tourn_size, self.rng, stable_sort
                    )

            # Replace points array and re-initialize metric arrays
            self.points = new_points
            self.objective_values = np.empty((self.num_niches, new_pop_size))
            self.violations = np.zeros((self.num_niches, new_pop_size))
            self.penalized_objectives = np.empty((self.num_niches, new_pop_size))
            self.fitnesses = np.empty((self.num_niches, new_pop_size))
            self.is_noptimal = np.empty((self.num_niches, new_pop_size), dtype=np.bool_)
        
            self.pop_size = new_pop_size

        if new_parent_size != self.parent_size:
            self.parents = np.empty((self.num_niches, new_parent_size, self.problem.ndim))
            self.parent_size = new_parent_size

    def evolve(self, elite_count, tourn_count, tourn_size, mutation_prob, 
               mutation_sigma, crossover_prob, niche_elitism, rng, stable_sort):
        """
        Performs one generation of evolution: selection, crossover, and mutation.
        """
        # 1. Select parents
        parents = self._select_parents(
            elite_count, tourn_count, tourn_size, rng, stable_sort
        )
        
        # 2. Generate offspring
        offspring = self._generate_offspring(
            parents, crossover_prob, mutation_prob, mutation_sigma, niche_elitism, rng,
        )
        self.points = offspring

        # 3. Enforce constraints
        self._apply_bounds()
        self._apply_integrality()

    def evaluate_and_update(self, noptimal_slack):
        """
        Evaluates objective, fitness, and updates the best-known solutions.
        """
        # Evaluate objectives and apply penalties
        for i in range(self.num_niches):
            self.objective_values[i], self.violations[i] = self.problem.evaluate(self.points[i])
        # TODO: Update penalty factor
        self.penalized_objectives = self.objective_values + self.violations # Assumes factor=1
        
        # Update global optimum
        self._update_global_best()
        
        # Determine near-optimality based on the new global best
        if self.problem.maximize:
            self.noptimal_threshold = self.global_best_value / noptimal_slack
            self.is_noptimal[:] = self.objective_values > self.noptimal_threshold
        else:
            self.noptimal_threshold = self.global_best_value * noptimal_slack
            self.is_noptimal[:] = self.objective_values < self.noptimal_threshold

        # Evaluate fitness based on diversity
        self._evaluate_fitness()

    def get_best_solutions(self) -> dict:
        """
        Finds the best individual from each niche.
        """
        if self.problem.maximize:
            opt_idx = np.argmax(self.objective_values[0])
        else:
            opt_idx = np.argmin(self.objective_values[0])
            
        best_indices = [opt_idx]
        for i in range(1, self.num_niches):
            # Prioritize n-optimal, feasible solutions
            noptimal_feasible_mask = self.is_noptimal[i] & (self.violations[i] == 0)
            if np.any(noptimal_feasible_mask):
                best_indices.append(np.argmax(self.fitnesses[i][noptimal_feasible_mask]))
            else: # Fallback to best fitness if none are n-optimal
                best_indices.append(np.argmax(self.fitnesses[i]))
        
        niche_ids = np.arange(self.num_niches)
        return {
            'points': self.points[niche_ids, best_indices, :],
            'fitness': self.fitnesses[niche_ids, best_indices],
            'objective': self.objective_values[niche_ids, best_indices],
            'is_noptimal': self.is_noptimal[niche_ids, best_indices],
        }

    def get_best_objective_per_niche(self) -> np.ndarray:
        """
        Gets the best objective value for each niche.
        """
        if self.problem.maximize:
            return np.max(self.objective_values, axis=1)
        else:
            return np.min(self.objective_values, axis=1)

    def _select_parents(self, elite_count, tourn_count, tourn_size, rng, stable):
        """
        Selects parents for the next generation.
        """
        _select_parents(
            self.parents,
            self.points, 
            self.penalized_objectives, 
            self.fitnesses, 
            self.is_noptimal, 
            self.problem.maximize, 
            elite_count, 
            tourn_count, 
            tourn_size, 
            rng, 
            stable, 
            )

    def _generate_offspring(self, parents, cx_prob, mut_prob, mut_sigma, niche_elitism, rng):
        """
        Generates offspring from parents via cloning, crossover, and mutation.
        """
        # Clone parents to form the base for the new generation
        num_parents = parents.shape[1]
        num_clones = self.pop_size // num_parents
        remainder = self.pop_size % num_parents
        
        offspring = np.repeat(parents, num_clones, axis=1)
        if remainder > 0:
            offspring = np.concatenate((offspring, parents[:, :remainder]), axis=1)
        
        # Crossover
        crossover.crossover_population(offspring, cx_prob, self.cx_func, rng)
        
        # Mutation
        sigma_vals = mut_sigma * (self.problem.upper_bounds - self.problem.lower_bounds)
        mutation.mutate_gaussian_population_mixed(
            population=offspring,
            sigma=sigma_vals,
            indpb=mut_prob,
            rng=rng,
            integrality=self.problem.integrality,
            boolean_mask=self.problem.boolean_mask
        )

        # Elitism: preserve best individuals
        if niche_elitism:
            offspring[0, 0, :] = self.global_best_point
            if self.num_niches > 1:
                # This part could be enhanced, for now, we just preserve the best from each parent set
                for i in range(1, self.num_niches):
                    best_parent_idx = np.argmax(self.fitnesses[i])
                    offspring[i, 0, :] = self.points[i, best_parent_idx, :]
                    
        return offspring

    def _update_global_best(self):
        """
        Checks for and updates the global best solution found so far.
        """
        feasible_mask = self.violations == 0
        if not np.any(feasible_mask):
            return

        if self.problem.maximize:
            current_best_val = np.max(self.objective_values[feasible_mask])
            if current_best_val > self.global_best_value:
                self.global_best_value = current_best_val
                idx = np.unravel_index(np.argmax(self.objective_values), self.objective_values.shape)
                self.global_best_point = self.points[idx]
        else:
            current_best_val = np.min(self.objective_values[feasible_mask])
            if current_best_val < self.global_best_value:
                self.global_best_value = current_best_val
                idx = np.unravel_index(np.argmin(self.objective_values), self.objective_values.shape)
                self.global_best_point = self.points[idx]
    
    def _evaluate_fitness(self):
        """
        Calculates fitness for each individual based on distance to other niche centroids.
        """
        _evaluate_fitness(self.fitnesses, self.centroids, self.points)
    
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
                np.round(self.points[:, :, k], out=self.points[:, :, k])

@njit
def _initialize_randomly(points, start_idx, end_idx, lb, ub, rng):
    for i in range(start_idx, end_idx):
        for j in range(points.shape[1]):
            for k in range(points.shape[2]):
                points[i, j, k] = rng.uniform(lb[k], ub[k])   

@njit
def _apply_bounds(points, lb, ub):
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            for k in range(points.shape[2]):
                points[i, j, k] = min(ub[k], max(lb[k], points[i, j, k]))

@njit
def _select_parents(parents, points, objectives, fitnesses, is_noptimal, maximize, elite_count, tourn_count, tourn_size, rng, stable):
    """
    Selects parents for the next generation.
    """
    # Niche 0 (optimization) selects on objective value
    parents[0] = selection.selection(
        points[0], objectives[0], maximize, elite_count, tourn_count, tourn_size, rng, stable
    )
    
    # Other niches (diversity) select on fitness, with objective as fallback
    for i in range(1, parents.shape[0]):
        parents[i] = selection.selection_with_fallback(
            points[i], fitnesses[i], is_noptimal[i], objectives[i], maximize, elite_count, tourn_count, tourn_size, rng, stable
        )

@njit 
def _evaluate_fitness(fitnesses, centroids, points):
    _find_centroids(centroids, points)
    fit_metrics.evaluate_fitness_dist_to_centroids(fitnesses, points, centroids)


@njit
def _find_centroids(centroids, points):
    """
    Calculates the geometric center (centroid) of each niche.
    """
    centroids[:,:] = 0.0
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            for k in range(points.shape[2]):
                centroids[i, k] += points[i, j, k]
    centroids /= points.shape[1]

@njit
def _add_niche_to_array(old_array, num_niches):
    ndim = old_array.ndim
    if ndim == 3:
        new_array = np.empty(
            (num_niches, 
             old_array.shape[1], 
             old_array.shape[2],
             ), 
            dtype=old_array.dtype)
        for i in range(old_array.shape[0]):
            for j in range(old_array.shape[1]):
                for k in range(old_array.shape[2]):
                    new_array[i, j, k] = old_array[i, j, k]
    elif ndim == 2:
        new_array = np.empty(
            (num_niches, 
             old_array.shape[1], 
             ), 
            dtype=old_array.dtype)
        for i in range(old_array.shape[0]):
            for j in range(old_array.shape[1]):
                    new_array[i, j] = old_array[i, j]
    else: 
        raise Exception
    return new_array

@njit
def clone(target, start_pop):
    nindividuals = start_pop.shape[1]
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            jn = j%nindividuals
            for k in range(target.shape[2]):
                target[i, j, k] = start_pop[i, jn, k]