import numpy as np
from numba import njit
from scipy.spatial import Delaunay
import warnings 

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
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
        self.num_niches = INT(num_niches)
        self.pop_size = INT(pop_size)
        self.parent_size = INT(0)
        self.unselfish_niche_fit = FLOAT(0.0)
        self.rng = rng
        
        # Population data arrays
        self.points = np.empty((num_niches, pop_size, problem.ndim), dtype=FLOAT)
        self.objective_values = np.empty((num_niches, pop_size), dtype=FLOAT)
        self.violations = np.zeros((num_niches, pop_size), dtype=FLOAT)
        self.penalized_objectives = np.empty((num_niches, pop_size), dtype=FLOAT)
        self.fitnesses = np.empty((num_niches, pop_size), dtype=FLOAT)
        self.is_noptimal = np.empty((num_niches, pop_size), dtype=np.bool_)
        self.centroids = np.empty((num_niches, problem.ndim), dtype=FLOAT)
        self.niche_elites = np.empty((num_niches - 1, 1, problem.ndim), dtype=FLOAT)
        
        # Overall best found
        self.current_optima = np.empty((num_niches, problem.ndim), dtype=FLOAT)
        self.current_optima_obj = np.empty((num_niches), dtype=FLOAT)
        self.current_optima_pob = np.empty((num_niches), dtype=FLOAT)
        self.current_optima_fit = np.empty((num_niches), dtype=FLOAT)
        self.current_optima_nop = np.empty((num_niches), dtype=np.bool_)
        self.noptimal_threshold = -np.inf if problem.maximize else np.inf

        self.current_optima[0] = problem.known_optimum_point
        self.current_optima_obj[0] = problem.known_optimum_value

        # Auxiliary functions
        self.cx_func = crossover._cx_two_point if problem.ndim > 2 else crossover._cx_one_point

    def populate(self):
        """
        populates the population points with a uniform distribution
        """
        self._populate_randomly(INT(0), self.num_niches)
        self._apply_integrality()
        self._apply_bounds()

    def _populate_randomly(self, start_idx, end_idx):
        """Helper to populate a slice of niches with random points."""
        _populate_randomly(
            self.points, 
            start_idx, 
            end_idx, 
            self.problem.lower_bounds,
            self.problem.upper_bounds,
            self.rng,
        )

    def heuristic_add_niches(self, num_new_niches: int):
        """
        Uses delaunay triangulation to place the new niches inside 
        the convex hull of existing points. (Not tractable above ~8 dim)
        """
        if num_new_niches == 0:
            return
        if num_new_niches < 0:
            raise ValueError("Number of new niches must be positive")
        if self.problem.ndim > 8:
            warnings.warn(f"problem ndim is {self.problem.ndim}. This is likely prohibitively"\
                          "heuristic niche adding. Try non-heuristic niche adding", RuntimeWarning)

        num_old_niches = self.num_niches
        num_total_niches = num_old_niches + num_new_niches

        self.resize(niches = num_total_niches)

        if num_old_niches >= self.problem.ndim + 1:
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
                    self._populate_randomly(niche_idx, niche_idx + 1)
        else: 
            raise RuntimeError("Not enough niches to triangulate. Try non-heuristic niche adding")

        self.num_niches = num_total_niches
        self._apply_integrality()
        self._apply_bounds()

    def add_niches(self, num_new_niches: int):
        """
        Adds niches to the population. 
        """
        if num_new_niches == 0:
            return
        if num_new_niches < 0:
            raise ValueError("Number of new niches must be positive")

        num_old_niches = self.num_niches
        num_total_niches = num_old_niches + num_new_niches
        self.resize(niches = num_total_niches)
        
        self._populate_randomly(num_old_niches, num_total_niches)
        
        self.num_niches = num_total_niches
        self._apply_integrality()
        self._apply_bounds()

    def resize(
            self, 
            num_niches: int = None, 
            pop_size: int = None, 
            parent_size:int = None, 
            stable_sort: bool = None
            ):
        """
        Resizes the population of each niche to a new size.
        """
        if num_niches is not None: 
            self._resize_niche_size(num_niches)
        if pop_size is not None:
            if stable_sort is None:
                warnings.warn("resizing pop size: no stable sort argument provided. Assuming True", RuntimeWarning)
                stable_sort = True
            self._resize_pop_size(pop_size, stable_sort)
        if parent_size is not None: 
            self._resize_parent_size(parent_size)

    def evolve(
            self, 
            elite_count: int, 
            tourn_count: int, 
            tourn_size: int, 
            mutation_prob: float, 
            mutation_sigma: float, 
            crossover_prob: float, 
            niche_elitism: str|None, 
            rng: np.random._generator.Generator, 
            stable_sort: bool,
            ):
        """
        Performs one generation of evolution: selection, crossover, and mutation.
        """
        # 1. Select parents
        self._select_parents(
            elite_count, tourn_count, tourn_size, rng, stable_sort
        )
        
        # 2. Generate offspring
        offspring = self._generate_offspring(
            self.parents, crossover_prob, mutation_prob, mutation_sigma, niche_elitism, rng,
        )
        ## TODO: can we do this in-place to begin with?
        njit_deepcopy(self.points, offspring)

        # 3. Enforce constraints
        self._apply_bounds()
        self._apply_integrality()

    def evaluate_and_update(self, noptimal_slack, violation_factor):
        """
        Evaluates objective, fitness, and updates the best-known solutions.
        """
        # Evaluate objectives and apply penalties
        for i in range(self.num_niches):
            self.objective_values[i], self.violations[i] = self.problem.evaluate(self.points[i])
        
        self.penalized_objectives[:] = self.objective_values + self.violations * violation_factor
        self._evaluate_fitness()

        # Update global optimum
        self._update_optima(noptimal_slack)

        # Evaluate fitness
        self._evaluate_diversity()

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
        if niche_elitism == "unselfish":
            # rather than choosing the highest fitness individual from each niche (current noptima), we 
            # take either the current noptima or a previous set of noptima - whichever has the highest 
            # fitness measured to each other (rather than to the centroids)

            # container for fitness function evaluated on set of elites rather than centroids
            _fitness = np.empty((self.num_niches, 1))
            # evaluate fitness w.r.t. each other
            _evaluate_fitness(_fitness, self.current_optima, self.current_optima)
            # update niche_elites
            if np.mean(_fitness) > self.unselfish_niche_fit:
                self.unselfish_niche_fit = np.mean(_fitness)
                njit_deepcopy(self.niche_elites, self.current_optima[1:])

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
        offspring[0, 0, :] = self.current_optima[0]
        if niche_elitism == "selfish":
            # This part could be enhanced, for now, we just preserve the best from each parent set
            for i in range(1, self.num_niches):
                offspring[i, 0, :] = self.current_optima[i, :]
        elif niche_elitism == "unselfish":
            for i in range(1, self.num_niches):
                offspring[i, 0, :] = self.niche_elites[i-1]
                    
        return offspring

    def _update_optima(self, noptimal_slack):
        """
        Checks for and updates the global best solution found so far.
        Updates near-optima according to near-optimal slack
        """
        feasible_mask = self.violations == 0
        if not np.any(feasible_mask):
            return

        idx = None
        if self.problem.maximize:
            current_best_val = np.max(self.objective_values[feasible_mask])
            if current_best_val > self.current_optima_obj[0]:
                self.current_optima_obj[0] = current_best_val
                idx = np.unravel_index(np.argmax(self.penalized_objectives), self.objective_values.shape)
        else:
            current_best_val = np.min(self.objective_values[feasible_mask])
            if current_best_val < self.current_optima_obj[0]:
                self.current_optima_obj[0] = current_best_val
                idx = np.unravel_index(np.argmin(self.penalized_objectives), self.objective_values.shape)
        
        if idx is not None:
            self.current_optima[0, :] = self.points[idx]
            self.current_optima_obj[0] = self.objective_values[idx]
            self.current_optima_pob[0] = self.penalized_objectives[idx]
            self.current_optima_fit[0] = self.fitnesses[idx]
            # logically must be true but self.is_noptimal has not been calculated yet
            self.current_optima_nop[0] = True 

        if self.current_optima_obj[0] < 0:
            warnings.warn(
                "Negative optimal objective encountered. Optimum should be positive definite for mga slack logic."\
                "Consider using a large scalar offset.",
                RuntimeWarning
            )
        self.noptimal_threshold = _noptimal_threshold(self.current_optima_obj[0], noptimal_slack, self.problem.maximize)
        
        # Determine near-optimality based on the current optimum
        _evaluate_noptimality(self.is_noptimal, self.penalized_objectives, self.noptimal_threshold, self.problem.maximize)

        _idx = np.empty(self.points.shape[0], INT)
        for i in range(1, self.points.shape[0]):
            if feasible_mask[i].any():
                _best = -np.inf
                for j in range(self.points.shape[1]):
                    if self.fitnesses[i, j] > _best:
                        _best = self.fitnesses[i, j]
                        _idx = j
            elif self.problem.maximize:
                _best = -np.inf
                for j in range(self.points.shape[1]):
                    if self.penalized_objectives[i, j] > _best:
                        _best = self.penalized_objectives[i, j]
                        _idx = j
            else:
                _best = np.inf
                for j in range(self.points.shape[1]):
                    if self.penalized_objectives[i, j] < _best:
                        _best = self.penalized_objectives[i, j]
                        _idx = j
            self.current_optima[i, :] = self.points[i, j, :]
            self.current_optima_obj[i] = self.objective_values[i, _idx]
            self.current_optima_pob[i] = self.penalized_objectives[i, _idx]
            self.current_optima_fit[i] = self.fitnesses[i, _idx]
            self.current_optima_nop[i] = self.is_noptimal[i, _idx]

    def _evaluate_fitness(self):
        """
        Calculates fitness for each individual based on distance to other niche centroids.
        """
        _evaluate_fitness(self.fitnesses, self.centroids, self.points)

    def _evaluate_diversity(self):
        """
        calculate vesa, shannon, etc.
        """
        pass
    
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

    def _resize_niche_size(self, new_niche_size: int):
        if new_niche_size != self.num_niches:
            new_niche_size = INT(new_niche_size)
            self.points = _add_niche_to_array(self.points, new_niche_size)
            self.objective_values = _add_niche_to_array(self.objective_values, new_niche_size)
            self.violations = _add_niche_to_array(self.violations, new_niche_size)
            self.penalized_objectives = _add_niche_to_array(self.penalized_objectives, new_niche_size)
            self.fitnesses = _add_niche_to_array(self.fitnesses, new_niche_size)
            self.is_noptimal = _add_niche_to_array(self.is_noptimal, new_niche_size)
            self.centroids = _add_niche_to_array(self.centroids, new_niche_size)
            self.niche_elites = _add_niche_to_array(self.niche_elites, new_niche_size)

            self.current_optima = _add_niche_to_array(self.current_optima, new_niche_size)
            self.current_optima_obj = _add_niche_to_array(self.current_optima_obj, new_niche_size)
            self.current_optima_pob = _add_niche_to_array(self.current_optima_pob, new_niche_size)
            self.current_optima_fit = _add_niche_to_array(self.current_optima_fit, new_niche_size)
            self.current_optima_nop = _add_niche_to_array(self.current_optima_nop, new_niche_size)

            self.unselfish_niche_fit = FLOAT(0.0)
            self.num_niches = new_niche_size


    def _resize_pop_size(self, new_pop_size: int, stable_sort: bool):
        if new_pop_size != self.pop_size:
            # Create new arrays with the target size
            new_points = np.empty((self.num_niches, new_pop_size, self.problem.ndim))
            
            if new_pop_size > self.pop_size:
                _clone(new_points, self.points)

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

    def _resize_parent_size(self, new_parent_size):
        if new_parent_size != self.parent_size:
            self.parents = np.empty((self.num_niches, new_parent_size, self.problem.ndim))
            self.parent_size = INT(new_parent_size)

#%%

@njit
def _populate_randomly(points, start_idx, end_idx, lb, ub, rng):
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
    selection.selection(
        parents[0], points[0], objectives[0], maximize, elite_count, tourn_count, tourn_size, rng, stable
    )
    
    # Other niches (diversity) select on fitness, with objective as fallback
    for i in range(1, parents.shape[0]):
        selection.selection_with_fallback(
            parents[i], points[i], fitnesses[i], is_noptimal[i], objectives[i], maximize, elite_count, tourn_count, tourn_size, rng, stable
        )

@njit 
def _evaluate_fitness(fitnesses, centroids, points):
    """
    evaluate fitnesses of populations points
    """
    _find_centroids(centroids, points)
    fit_metrics.evaluate_fitness_dist_to_centroids(fitnesses, points, centroids)

@njit
def _evaluate_noptimality(is_noptimal, objective, threshold, maximize):
    """
    evaluate noptimality of evaluated points
    """
    if maximize: 
        for i in range(is_noptimal.shape[0]):
            for j in range(is_noptimal.shape[1]):
                is_noptimal[i, j] = objective[i, j] > threshold
    else: 
        for i in range(is_noptimal.shape[0]):
            for j in range(is_noptimal.shape[1]):
                is_noptimal[i, j] = objective[i, j] < threshold

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
            dtype=old_array.dtype
        )
        for i in range(old_array.shape[0]):
            for j in range(old_array.shape[1]):
                for k in range(old_array.shape[2]):
                    new_array[i, j, k] = old_array[i, j, k]
    elif ndim == 2:
        new_array = np.empty(
            (num_niches, 
             old_array.shape[1], 
             ), 
            dtype=old_array.dtype
        )
        for i in range(old_array.shape[0]):
            for j in range(old_array.shape[1]):
                    new_array[i, j] = old_array[i, j]
    elif ndim == 1: 
        new_array = np.empty(
            (num_niches, ),
            dtype=old_array.dtype
        )
        for i in range(old_array.shape[0]):
            new_array[i] = old_array[i]
    else: 
        raise Exception
    return new_array

@njit
def _clone(target, start_pop):
    nindividuals = start_pop.shape[1]
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            jn = j%nindividuals
            for k in range(target.shape[2]):
                target[i, j, k] = start_pop[i, jn, k]

@njit
def _noptimal_threshold(optimal_obj, slack, maximize):
    if maximize:
        return optimal_obj * (1-(slack-1))
    return optimal_obj * slack

@njit 
def njit_deepcopy(new, old):
    flat_new = new.ravel()
    flat_old = old.ravel()
    
    for i in range(flat_old.shape[0]):
        flat_new[i] = flat_old[i]