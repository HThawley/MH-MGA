import numpy as np
from numba.types import npy_rng

from mga.commons.constants import JIT_ENABLED
from mga.commons.numba_overload import njit, jitclass
from mga.commons.types import nbint, npintp, nbfloat, npfloat, boolean
from mga.problem_definition import OptimizationProblem
from mga.operators import selection, crossover, mutation
from mga.metrics import fitness as fit_metrics
from mga.metrics import diversity

if JIT_ENABLED:
    spec = [
        ('num_niches', nbint),
        ('pop_size', nbint),
        ('ndim', nbint),
        ('parent_size', nbint),
        ('unselfish_niche_fitness_threshold', nbfloat),
        ('stable_sort', boolean),
        ('include_obj_in_fitness', boolean),

        ('integrality', boolean[:]),
        ('booleanality', boolean[:]),
        ('maximize', boolean),
        ('scaling_in_obj_func', boolean),
        ('lower_bounds', nbfloat[:]),
        ('scaled_lower_bounds', nbfloat[:]),
        ('upper_bounds', nbfloat[:]),
        ('scaled_upper_bounds', nbfloat[:]),
        ('problem_loaded', boolean),
        ('rng', npy_rng),
        ('is_continuous_space', boolean),

        # Main population arrays
        ('points', nbfloat[:, :, :]),
        ('scaled_points', nbfloat[:, :, :]),
        ('raw_objectives', nbfloat[:, :]),
        ('violations', nbfloat[:, :]),
        ('penalized_objectives', nbfloat[:, :]),
        ('feasible_mask', boolean[:, :]),
        ('fitnesses', nbfloat[:, :]),
        ('noptimal_mask', boolean[:, :]),
        ('scaled_centroids', nbfloat[:, :]),
        ('niche_elites', nbfloat[:, :, :]),
        ('parents', nbfloat[:, :, :]),

        # Optima tracking
        ('optima_points', nbfloat[:, :]),
        ('optima_scaled_points', nbfloat[:, :]),
        ('optima_raw_objectives', nbfloat[:]),
        ('optima_violations', nbfloat[:]),
        ('optima_penalized_objectives', nbfloat[:]),
        ('optima_fitnesses', nbfloat[:]),
        ('optima_noptimal_mask', boolean[:]),
        ('noptimal_threshold', nbfloat),

        # Hyperparameters
        # (Defined individually in spec as they are assigned in update_hyperparameters)
        ('elite_count', nbint),
        ('tourn_count', nbint),
        ('tourn_size', nbint),
        ('mutation_prob', nbfloat[:]),
        ('mutation_sigma', nbfloat[:]),
        ('crossover_prob', nbfloat[:]),
        ('noptimal_rel', nbfloat),
        ('noptimal_abs', nbfloat),
        ('violation_factor', nbfloat),
        ('niche_elitism', nbint),
        ('mutation_scaler', nbfloat[:]),
        ('space_scaler', nbfloat[:]),
        ('objective_scaler', nbfloat),
        ('current_mutation_prob', nbfloat),
        ('mutation_sigma_inst', nbfloat),
        ('current_crossover_prob', nbfloat),

        # Metrics
        ('vesa', nbfloat),
        ('shannon', nbfloat),
        ('stds', nbfloat[:]),
        ('variances', nbfloat[:]),
        ('mean_fitness', nbfloat),
    ]
else:
    spec = []


@jitclass(spec)
class Population:
    """
    Manages the state and evolution of the entire population across all niches.
    """

    def __init__(
        self,
        num_niches: int,
        pop_size: int,
        ndim: int,
        stable_sort: bool,
        rng: npy_rng,
        include_obj_in_fitness: bool
    ):
        self.num_niches = nbint(num_niches)
        self.pop_size = nbint(pop_size)
        self.ndim = nbint(ndim)
        self.parent_size = nbint(0)
        self.unselfish_niche_fitness_threshold = nbfloat(0.0)
        self.stable_sort = stable_sort
        self.rng = rng
        self.include_obj_in_fitness = include_obj_in_fitness

        self.integrality = np.empty(ndim, np.bool_)
        self.booleanality = np.empty(ndim, np.bool_)
        self.maximize = False
        self.scaling_in_obj_func = False
        self.lower_bounds = np.empty(ndim, dtype=npfloat)
        self.scaled_lower_bounds = np.empty(ndim, dtype=npfloat)
        self.upper_bounds = np.empty(ndim, dtype=npfloat)
        self.scaled_upper_bounds = np.empty(ndim, dtype=npfloat)
        self.problem_loaded = False

        # Population data arrays
        self.points = np.empty((self.num_niches, self.pop_size, self.ndim), dtype=npfloat)
        self.scaled_points = np.empty((self.num_niches, self.pop_size, self.ndim), dtype=npfloat)
        self.raw_objectives = np.empty((self.num_niches, self.pop_size), dtype=npfloat)
        self.violations = np.zeros((self.num_niches, self.pop_size), dtype=npfloat)
        self.penalized_objectives = np.empty((self.num_niches, self.pop_size), dtype=npfloat)
        self.feasible_mask = np.empty((self.num_niches, self.pop_size), dtype=np.bool_)
        self.fitnesses = np.empty((self.num_niches, self.pop_size), dtype=npfloat)
        self.noptimal_mask = np.empty((self.num_niches, self.pop_size), dtype=np.bool_)
        self.niche_elites = np.empty((self.num_niches - 1, 1, self.ndim), dtype=npfloat)
        self.parents = np.empty((self.num_niches, 0, self.ndim), dtype=npfloat)
        if self.include_obj_in_fitness:
            self.scaled_centroids = np.empty((self.num_niches, self.ndim+1), dtype=npfloat)
        else:
            self.scaled_centroids = np.empty((self.num_niches, self.ndim), dtype=npfloat)

        # Overall best found
        self.optima_points = np.empty((self.num_niches, self.ndim), dtype=npfloat)
        self.optima_scaled_points = np.empty((self.num_niches, self.ndim), dtype=npfloat)
        self.optima_raw_objectives = np.empty((self.num_niches), dtype=npfloat)
        self.optima_violations = np.empty((self.num_niches), dtype=npfloat)
        self.optima_penalized_objectives = np.empty((self.num_niches), dtype=npfloat)
        self.optima_fitnesses = np.empty((self.num_niches), dtype=npfloat)
        self.optima_noptimal_mask = np.empty((self.num_niches), dtype=np.bool_)
        self.noptimal_threshold = 0.0

        self.elite_count = 0
        self.tourn_count = 0
        self.tourn_size = 0
        self.mutation_prob = np.empty(2, dtype=npfloat)
        self.mutation_sigma = np.empty(2, dtype=npfloat)
        self.crossover_prob = np.empty(2, dtype=npfloat)
        self.niche_elitism = 0
        self.noptimal_rel = 0.0
        self.noptimal_abs = 0.0
        self.violation_factor = 0.0
        self.mutation_scaler = np.empty(ndim, dtype=npfloat)
        self.space_scaler = np.empty(ndim, dtype=npfloat)
        self.objective_scaler = 1.0

    def _load_optimum(
            self,
            point,  # np.NDArray[float]
            objective: float,
            violation: float,
            violation_factor: float,
    ):
        if point.ndim == 0 or point.size == 0:
            return
        if not point.ndim == 1:
            raise ValueError("optimum point should be 1D")
        if not point.shape[0] == self.ndim:
            raise ValueError(f"optimum point should have length {self.ndim}. Got: {point.shape[0]}")

        self.optima_points[0, :] = point
        self.optima_raw_objectives[0] = objective
        self.optima_violations[0] = violation
        self.optima_penalized_objectives[0] = objective + violation_factor * violation
        self.optima_noptimal_mask[0] = False  # will be updated later

    def initialize_population(
            self,
            points,  # : NDArray[float]
    ):
        """
        populates the population points with a uniform distribution
        """
        if not self.problem_loaded:
            raise RuntimeError("Problem not loaded. Load problem before populating.")

        if points.ndim == 0 or points.size == 0:
            self._populate_randomly(0, self.num_niches)
        else:
            if points.ndim == 1:
                if not points.shape[0] == self.ndim:
                    raise ValueError(f"1D 'points' should have shape ({self.ndim},). Got {points.shape}.")
                points = points.reshape(1, 1, self.ndim)
            elif points.ndim == 2:
                if not (points.shape[0] == self.pop_size and points.shape[1] == self.ndim):
                    raise ValueError(
                        f"2D 'points' should have shape (pop_size={self.pop_size}, ndim={self.ndim}). Got {points}."
                        " If you are supplying (num_niches, ndim), try reshaping to (num_niches, 1, ndim)"
                    )
                points = points.reshape(1, self.pop_size, self.ndim)
            elif points.ndim == 3:
                if not (points.shape[0] == self.num_niches
                        and points.shape[1] == self.pop_size
                        and points.shape[2] == self.ndim):
                    raise ValueError(
                        f"3D 'points' should have shape (num_niches={self.num_niches}, pop_size={self.pop_size},"
                        f"ndim={self.ndim}). Got {points.shape}"
                    )
            else:
                raise ValueError(f"'points' should be 1D, 2D, or 3D array. Got {points.ndim}D array.")
            _clone(self.points, points)

        self.points[0, 0, :] = self.optima_points[0]  # inject known optimum

        self._apply_integrality()
        self._apply_bounds()

    def _populate_randomly(self, start_idx, end_idx):
        """Helper to populate a slice of niches with random points."""
        for i in range(start_idx, end_idx):
            for j in range(self.pop_size):
                for k in range(self.ndim):
                    self.points[i, j, k] = self.rng.uniform(self.lower_bounds[k], self.upper_bounds[k])

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
        self.resize(num_niches=num_total_niches)

        self._populate_randomly(num_old_niches, num_total_niches)

        self.num_niches = num_total_niches
        self._apply_integrality()
        self._apply_bounds()

    def resize(
        self,
        num_niches: int = -1,
        pop_size: int = -1,
        parent_size: int = -1,
    ):
        """
        Resizes the population of each niche to a new size.
        """
        if num_niches != -1:
            self._resize_niche_size(num_niches)
        if pop_size != -1:
            self._resize_pop_size(pop_size)
        if parent_size != -1:
            self._resize_parent_size(parent_size)

    def update_hyperparameters(
        self,
        elite_count: int,
        tourn_count: int,
        tourn_size: int,
        mutation_prob: np.ndarray[float],
        mutation_sigma: np.ndarray[float],
        crossover_prob: np.ndarray[float],
        niche_elitism: int,
        noptimal_rel: float,
        noptimal_abs: float,
        violation_factor: float,
        mutation_scaler: np.ndarray[float],
        space_scaler: np.ndarray[float],
        objective_scaler: float,
    ):
        self.elite_count = nbint(elite_count)
        self.tourn_count = nbint(tourn_count)
        self.tourn_size = nbint(tourn_size)
        self.mutation_prob[:] = mutation_prob.astype(nbfloat)
        self.mutation_sigma[:] = mutation_sigma.astype(nbfloat)
        self.crossover_prob[:] = crossover_prob.astype(nbfloat)
        self.niche_elitism = nbint(niche_elitism)
        self.noptimal_rel = nbfloat(noptimal_rel)
        self.noptimal_abs = nbfloat(noptimal_abs)
        self.violation_factor = nbfloat(violation_factor)
        self.mutation_scaler[:] = mutation_scaler.astype(nbfloat)
        self.space_scaler[:] = space_scaler.astype(nbfloat)
        self.objective_scaler = nbfloat(objective_scaler)

        self._rescale_bounds()

    def select_parents(self):
        """
        Selects parents for the next generation.
        """
        # Niche 0 (optimization) selects on objective value
        selection.selection(
            self.parents[0],
            self.points[0],
            self.penalized_objectives[0],
            self.maximize,
            self.elite_count,
            self.tourn_count,
            self.tourn_size,
            self.rng,
            self.stable_sort
        )

        # Other niches (diversity) select on fitness, with objective as fallback
        for i in range(1, self.num_niches):
            selection.selection_with_fallback(
                self.parents[i],
                self.points[i],
                self.fitnesses[i],
                self.noptimal_mask[i],
                self.penalized_objectives[i],
                self.maximize,
                self.elite_count,
                self.tourn_count,
                self.tourn_size,
                self.rng,
                self.stable_sort,
            )

    def generate_offspring(self):
        """
        Generates offspring from parents via cloning, crossover, and mutation.
        """
        if self.niche_elitism == 2:
            # rather than choosing the highest fitness individual from each niche (current noptima), we
            # take either the current noptima or a previous set of noptima - whichever has the highest
            # fitness measured to each other (rather than to the centroids)

            # container for fitness function evaluated on set of elites rather than centroids
            _fitness = np.empty((self.num_niches, 1))
            # evaluate fitness w.r.t. each other
            fit_metrics.evaluate_fitness_dist_to_centroids(
                _fitness,
                np.atleast_3d(self.optima_scaled_points).transpose(1, 0, 2),
                self.optima_scaled_points
            )

            # update niche_elites
            if np.mean(_fitness) > self.unselfish_niche_fitness_threshold:
                self.unselfish_niche_fitness_threshold = np.mean(_fitness)
                njit_deepcopy(self.niche_elites, self.optima_points[1:])

        _clone(self.points, self.parents)

        # Crossover
        if self.ndim > 2:
            crossover.crossover_population(
                points=self.points,
                crossover_prob=self.current_crossover_prob,
                cx_func=crossover._cx_two_point,
                rng=self.rng,
            )
        else:
            crossover.crossover_population(
                points=self.points,
                crossover_prob=self.current_crossover_prob,
                cx_func=crossover._cx_one_point,
                rng=self.rng,
            )

        # Mutation
        sigma_vals = self.mutation_sigma_inst * self.mutation_scaler
        if self.is_continuous_space:
            mutation.mutate_gaussian_population_float(
                points=self.points,
                mutation_sigma=sigma_vals,
                mutation_prob=self.current_mutation_prob,
                rng=self.rng,
            )
        else:
            mutation.mutate_gaussian_population_mixed(
                points=self.points,
                mutation_sigma=sigma_vals,
                mutation_prob=self.current_mutation_prob,
                rng=self.rng,
                integrality=self.integrality,
                booleanality=self.booleanality,
            )

        # Elitism: preserve best individuals
        self.points[0, 0, :] = self.optima_points[0]
        if self.niche_elitism == 1:
            # This part could be enhanced, for now, we just preserve the best from each parent set
            for i in range(1, self.num_niches):
                if self.optima_noptimal_mask[i]:  # only preserve if noptimal
                    self.points[i, 0, :] = self.optima_points[i, :]
                else:  # otherwise duplicate the current optimum
                    self.points[i, 0, :] = self.optima_points[0, :]
        elif self.niche_elitism == 2:
            for i in range(1, self.num_niches):
                self.points[i, 0, :] = self.niche_elites[i - 1]

        self._apply_bounds()
        self._scale_points()

    def track_optima(self):
        """
        Checks for and updates the global best solution found so far.
        Updates near-optima according to near-optimal slack
        """

        self._update_feasible_mask()

        if np.any(self.feasible_mask):
            gi, gj = self._find_global_best_idx()
        else:
            if self.maximize:
                gi, gj = _argmax_2d(self.penalized_objectives)
            else:
                gi, gj = _argmin_2d(self.penalized_objectives)

        if gi != -1:
            self.optima_points[0, :] = self.points[gi, gj]
            self.optima_scaled_points[0, :] = self.scaled_points[gi, gj]
            self.optima_raw_objectives[0] = self.raw_objectives[gi, gj]
            self.optima_violations[0] = self.violations[gi, gj]
            self.optima_penalized_objectives[0] = self.penalized_objectives[gi, gj]
            self.optima_fitnesses[0] = self.fitnesses[gi, gj]
            # logically must be true but self.noptimal_mask has not been calculated yet
            self.optima_noptimal_mask[0] = True

        self._update_noptimal_threshold()
        self._update_noptimal_mask()

        self.feasible_mask *= self.noptimal_mask
        js = self._find_best_in_niche()
        for i in range(1, self.num_niches):
            self.optima_points[i, :] = self.points[i, js[i], :]
            self.optima_scaled_points[i, :] = self.scaled_points[i, js[i], :]
            self.optima_raw_objectives[i] = self.raw_objectives[i, js[i]]
            self.optima_violations[i] = self.violations[i, js[i]]
            self.optima_penalized_objectives[i] = self.penalized_objectives[i, js[i]]
            self.optima_fitnesses[i] = self.fitnesses[i, js[i]]
            self.optima_noptimal_mask[i] = self.noptimal_mask[i, js[i]]

    def evaluate_fitness(self):
        """
        Calculates fitness for each individual based on distance to other niche centroids.
        """
        self._find_scaled_centroids()
        if self.include_obj_in_fitness:
            fit_metrics.evaluate_fitness_dist_to_centroids_ext(
                self.fitnesses,
                self.scaled_points,
                self.scaled_centroids,
                self.raw_objectives,
                self.objective_scaler,
            )
        else:
            fit_metrics.evaluate_fitness_dist_to_centroids(
                self.fitnesses,
                self.scaled_points,
                self.scaled_centroids,
            )

    def evaluate_diversity(self):
        """
        Calculates and returns a vector of diversity metrics for the current population state.
        """
        scaled_nopt_points = self.optima_scaled_points[self.optima_noptimal_mask]

        # VESA
        if scaled_nopt_points.shape[0] >= self.ndim + 1:
            self.vesa = diversity.volume_estimation_by_shadow_addition(
                scaled_nopt_points, np.ones(scaled_nopt_points.shape[0], dtype=np.bool_)
            )
        else:
            self.vesa = 0.0

        # Shannon Index
        if self.scaling_in_obj_func:
            # TODO: figure out bounds in this case
            self.shannon = 0.0
        elif scaled_nopt_points.shape[0] >= 1:
            self.shannon = diversity.mean_of_shannon_of_projections(
                scaled_nopt_points,
                np.ones(scaled_nopt_points.shape[0], dtype=np.bool_),
                self.scaled_lower_bounds,
                self.scaled_upper_bounds,
            )
        else:
            self.shannon = 0.0

        # Fitness statistics
        if self.fitnesses.size > 0:
            self.stds = diversity.std(self.fitnesses, self.noptimal_mask)
            self.variances = diversity.var(self.fitnesses, self.noptimal_mask)
            self.mean_fitness = diversity.mean_of_fitness(self.fitnesses, self.noptimal_mask)
        else:
            self.stds = np.full(self.num_niches, np.inf)
            self.variances = np.full(self.num_niches, np.inf)
            self.mean_fitness = 0.0

    def dither_probabilities(self):
        self.current_mutation_prob = _dither(self.mutation_prob, self.rng)
        self.mutation_sigma_inst = _dither(self.mutation_sigma, self.rng)
        self.current_crossover_prob = _dither(self.crossover_prob, self.rng)

    def _apply_bounds(self):
        """
        Clips the population points to stay within the defined bounds.
        """
        for i in range(self.num_niches):
            for j in range(self.pop_size):
                for k in range(self.ndim):
                    self.points[i, j, k] = min(
                        self.upper_bounds[k],
                        max(self.lower_bounds[k], self.points[i, j, k])
                    )

    def _apply_integrality(self):
        """
        Rounds points for variables that are defined as integers.
        """
        for k in range(self.ndim):
            if self.integrality[k]:
                np.round(self.points[:, :, k], out=self.points[:, :, k])

    def _resize_niche_size(self, new_niche_size: int):
        if new_niche_size != self.num_niches:
            new_niche_size = nbint(new_niche_size)
            self.points = _add_niche_to_array(self.points, new_niche_size)
            self.scaled_points = _add_niche_to_array(self.scaled_points, new_niche_size)
            self.raw_objectives = _add_niche_to_array(self.raw_objectives, new_niche_size)
            self.violations = _add_niche_to_array(self.violations, new_niche_size)
            self.penalized_objectives = _add_niche_to_array(self.penalized_objectives, new_niche_size)
            self.feasible_mask = _add_niche_to_array(self.feasible_mask, new_niche_size)
            self.fitnesses = _add_niche_to_array(self.fitnesses, new_niche_size)
            self.noptimal_mask = _add_niche_to_array(self.noptimal_mask, new_niche_size)
            self.scaled_centroids = _add_niche_to_array(self.scaled_centroids, new_niche_size)
            self.niche_elites = _add_niche_to_array(self.niche_elites, new_niche_size - 1)
            self.parents = _add_niche_to_array(self.parents, new_niche_size)

            self.optima_points = _add_niche_to_array(self.optima_points, new_niche_size)
            self.optima_scaled_points = _add_niche_to_array(self.optima_scaled_points, new_niche_size)
            self.optima_raw_objectives = _add_niche_to_array(self.optima_raw_objectives, new_niche_size)
            self.optima_violations = _add_niche_to_array(self.optima_violations, new_niche_size)
            self.optima_penalized_objectives = _add_niche_to_array(self.optima_penalized_objectives, new_niche_size)
            self.optima_fitnesses = _add_niche_to_array(self.optima_fitnesses, new_niche_size)
            self.optima_noptimal_mask = _add_niche_to_array(self.optima_noptimal_mask, new_niche_size)

            self.unselfish_niche_fitness_threshold = nbfloat(0.0)
            self.num_niches = new_niche_size

    def _resize_pop_size(self, new_pop_size: int):
        if new_pop_size != self.pop_size:
            # Create new arrays with the target size
            new_points = np.empty((self.num_niches, new_pop_size, self.ndim))

            if new_pop_size > self.pop_size:
                _clone(new_points, self.points)

            else:  # new_pop_size < self.pop_size
                # Decrease size: select the best individuals to keep
                # Niche 0 (optimization) is selected based on objective value
                selection.selection(
                    new_points[0],
                    self.points[0],
                    self.penalized_objectives[0],
                    self.maximize,
                    self.elite_count,
                    self.tourn_count,
                    self.tourn_size,
                    self.rng,
                    self.stable_sort,
                )

                # Other niches (diversity) are selected based on fitness
                for i in range(1, self.num_niches):
                    selection.selection_with_fallback(
                        new_points[i],
                        self.points[i],
                        self.fitnesses[i],
                        self.noptimal_mask[i],
                        self.penalized_objectives[i],
                        self.maximize,
                        self.elite_count,
                        self.tourn_count,
                        self.tourn_size,
                        self.rng,
                        self.stable_sort,
                    )

            # Replace points array and re-initialize metric arrays
            self.points = new_points
            self.scaled_points = np.empty((self.num_niches, new_pop_size, self.ndim))
            if self.scaling_in_obj_func:
                self.scaled_points[:] = 0.0
            else:
                self._scale_points()
            self.raw_objectives = np.empty((self.num_niches, new_pop_size))
            self.violations = np.zeros((self.num_niches, new_pop_size))
            self.penalized_objectives = np.empty((self.num_niches, new_pop_size))
            self.feasible_mask = np.empty((self.num_niches, new_pop_size), dtype=np.bool_)
            self.fitnesses = np.empty((self.num_niches, new_pop_size))
            self.noptimal_mask = np.empty((self.num_niches, new_pop_size), dtype=np.bool_)

            self.pop_size = new_pop_size

    def _resize_parent_size(self, new_parent_size):
        if new_parent_size != self.parent_size:
            if self.parent_size > 0:
                new_parents = np.empty((self.num_niches, new_parent_size, self.ndim), dtype=npfloat)
                _clone(new_parents, self.parents)
                self.parents = new_parents
                self.parent_size = nbint(new_parent_size)
            else:
                self.parents = np.empty((self.num_niches, new_parent_size, self.ndim), dtype=npfloat)
                self.parent_size = nbint(new_parent_size)

    def _find_scaled_centroids(self):
        """
        Calculates the geometric center (centroid) of each niche.
        """
        self.scaled_centroids[:, :] = 0.0
        for i in range(self.num_niches):
            for j in range(self.pop_size):
                for k in range(self.ndim):
                    self.scaled_centroids[i, k] += self.scaled_points[i, j, k]

        if self.include_obj_in_fitness:
            for i in range(self.num_niches):
                for j in range(self.pop_size):
                    self.scaled_centroids[i, self.ndim] += _safe_divide(
                        self.raw_objectives[i, j], self.objective_scaler
                    )
        self.scaled_centroids /= self.points.shape[1]

    def _scale_points(self):
        """project points onto other space defined by space_scaler"""
        if self.scaling_in_obj_func:
            return
        for i in range(self.num_niches):
            for j in range(self.pop_size):
                for k in range(self.ndim):
                    self.scaled_points[i, j, k] = _safe_divide(self.points[i, j, k], self.space_scaler[k])

    def _rescale_bounds(self):
        for k in range(self.ndim):
            self.scaled_lower_bounds[k] = _safe_divide(self.lower_bounds[k], self.space_scaler[k])
            self.scaled_upper_bounds[k] = _safe_divide(self.upper_bounds[k], self.space_scaler[k])

    def _update_noptimal_mask(self):
        """
        evaluate noptimality of evaluated points
        noptimality is based on raw objective, not penalized objective
        """
        if self.maximize:
            for i in range(self.num_niches):
                for j in range(self.pop_size):
                    self.noptimal_mask[i, j] = self.penalized_objectives[i, j] > self.noptimal_threshold
        else:
            for i in range(self.num_niches):
                for j in range(self.pop_size):
                    self.noptimal_mask[i, j] = self.penalized_objectives[i, j] < self.noptimal_threshold

    def _update_noptimal_threshold(self):
        margin = abs(self.optima_penalized_objectives[0]) * (self.noptimal_rel) + self.noptimal_abs

        if self.maximize:
            self.noptimal_threshold = self.optima_penalized_objectives[0] - margin
        else:
            self.noptimal_threshold = self.optima_penalized_objectives[0] + margin

    def _update_feasible_mask(self):
        for i in range(self.num_niches):
            for j in range(self.pop_size):
                self.feasible_mask[i, j] = self.violations[i, j] == 0

    def _find_global_best_idx(self):
        """
        Returns flat index (niche, individual) of global best feasible point.
        If none feasible, returns (-1, -1).
        """
        if self.maximize:
            best_i, best_j = _argmax_with_mask_2d(
                self.penalized_objectives, self.feasible_mask, self.optima_penalized_objectives[0]
            )
        else:
            best_i, best_j = _argmin_with_mask_2d(
                self.penalized_objectives, self.feasible_mask, self.optima_penalized_objectives[0]
            )

        return best_i, best_j

    def _find_best_in_niche(self):
        """
        For each niche i>0, pick the representative index:
        1) best fitness among feasible & noptimal
        2) else best penalized objective
        Returns array of shape (num_niches,) giving chosen j for each niche.
        Niche 0 should be ignored by caller (global optimum already set).
        """
        out = np.full(self.num_niches, -1, dtype=npintp)

        for i in range(1, self.num_niches):
            if self.feasible_mask[i].sum() >= 1:
                # option 1: feasible & noptimal
                best_j = _argmax_with_mask(self.fitnesses[i], self.feasible_mask[i])

            else:
                # option 2: best penalized objective
                if self.maximize:
                    best_j = self.penalized_objectives[i].argmax()
                else:
                    best_j = self.penalized_objectives[i].argmin()

            out[i] = best_j
        return out


# %%
@njit
def _add_niche_to_array(old_array, num_niches):
    if num_niches < old_array.shape[0]:
        raise Exception
    # 3. Dynamically create the new shape
    new_shape = (num_niches,) + old_array.shape[1:]
    # 4. Create the new array
    new_array = np.empty(new_shape, dtype=old_array.dtype)
    # 5. Copy the old data into the new array using slicing
    # This is ndim-agnostic
    new_array[: old_array.shape[0]] = old_array

    return new_array


@njit
def _clone(target, parents):
    """
    fills a target array of shape (num_niches, pop_size, ndim) with clones
    of parents of shape (num_niches, parent_size, ndim) by repeating parents as needed.
    """
    nniches = parents.shape[0]
    nindividuals = parents.shape[1]
    for i in range(target.shape[0]):
        i_n = i % nniches
        for j in range(target.shape[1]):
            j_n = j % nindividuals
            for k in range(target.shape[2]):
                target[i, j, k] = parents[i_n, j_n, k]


@njit
def njit_deepcopy(new, old):
    flat_new = new.ravel()
    flat_old = old.ravel()

    for i in range(flat_old.shape[0]):
        flat_new[i] = flat_old[i]


@njit
def _argm_with_mask(array, mask, best, maximize):
    best_i = -1
    if maximize:
        for i in range(array.shape[0]):
            if mask[i]:
                val = array[i]
                if val > best:
                    best_i = i
                    best = val
    else:
        for i in range(array.shape[0]):
            if mask[i]:
                val = array[i]
                if val < best:
                    best_i = i
                    best = val
    return best_i


@njit
def _argmax_with_mask(array, mask):
    return _argm_with_mask(array, mask, -np.inf, True)


@njit
def _argmin_with_mask(array, mask):
    return _argm_with_mask(array, mask, np.inf, False)


@njit
def _argm_with_mask_2d(array, mask, best, maximize):
    best_i = -1
    best_j = -1
    if maximize:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if mask[i, j]:
                    val = array[i, j]
                    if val > best:
                        best_i = i
                        best_j = j
                        best = val
    else:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if mask[i, j]:
                    val = array[i, j]
                    if val < best:
                        best_i = i
                        best_j = j
                        best = val
    return best_i, best_j


@njit
def _argmax_with_mask_2d(array, mask, best=-np.inf):
    return _argm_with_mask_2d(array, mask, best, True)


@njit
def _argmin_with_mask_2d(array, mask, best=np.inf):
    return _argm_with_mask_2d(array, mask, best, False)


@njit
def _dither(bounds, rng):
    return rng.uniform(bounds[0], bounds[1])


@njit
def _safe_divide(
    num,
    denom,
    fail=0.0,
):
    """ Zero-safe division of two scalars """
    if denom == 0.0:
        return fail
    return num / denom


@njit
def _safe_divide_array(
    num,
    denom,
    fail=0.0,
):
    """ Zero-safe division of two arrays. """
    retarr = num.copy().ravel()
    denom_ravel = denom.ravel()
    for i in range(retarr.size):
        if denom_ravel[i] == 0.0:
            retarr[i] = fail
        else:
            retarr[i] /= denom_ravel[i]
    return retarr.reshape(num.shape)


@njit
def _argmax_2d(array):
    best_i = -1
    best_j = -1
    best = -np.inf
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            val = array[i, j]
            if val > best:
                best_i = i
                best_j = j
                best = val
    return best_i, best_j


@njit
def _argmin_2d(array):
    best_i = -1
    best_j = -1
    best = np.inf
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            val = array[i, j]
            if val < best:
                best_i = i
                best_j = j
                best = val
    return best_i, best_j


def load_problem_to_population(
        population: Population,
        problem: OptimizationProblem,
) -> None:
    population.integrality = problem.integrality
    population.booleanality = problem.booleanality
    population.maximize = problem.maximize
    population.scaling_in_obj_func = problem.return_scaled
    population.lower_bounds = problem.lower_bounds
    population.upper_bounds = problem.upper_bounds

    population.is_continuous_space = (population.integrality.sum() == 0)

    population.problem_loaded = True
