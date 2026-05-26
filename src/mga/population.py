import numpy as np
from numba.types import npy_rng

from mga.commons.constants import JIT_ENABLED
from mga.commons.numba_overload import njit, jitclass
from mga.commons.types import nbint, nbintp, npintp, nbfloat, npfloat, boolean
import mga.commons.utils as utils
from mga.problem_definition import OptimizationProblem
from mga.operators import selection, crossover, mutation
from mga.metrics import fitness
from mga.metrics import diversity

if JIT_ENABLED:
    spec = [
        ('num_niches', nbint),
        ('pop_size', nbint),
        ('ndim', nbint),
        ('parent_size', nbint),
        ('stable_sort', boolean),
        ('include_obj_in_fitness', boolean),
        ('pure_optimization', boolean),

        ('integrality', boolean[::1]),
        ('booleanality', boolean[::1]),
        ('maximize', boolean),
        ('scaling_in_obj_func', boolean),
        ('lower_bounds', nbfloat[::1]),
        ('scaled_lower_bounds', nbfloat[::1]),
        ('upper_bounds', nbfloat[::1]),
        ('scaled_upper_bounds', nbfloat[::1]),
        ('problem_loaded', boolean),
        ('rng', npy_rng),
        ('is_continuous_space', boolean),
        ('optimum_given', boolean),

        # Main population arrays
        ('points', nbfloat[:, :, ::1]),
        ('scaled_points', nbfloat[:, :, ::1]),
        ('raw_objectives', nbfloat[:, ::1]),
        ('violations', nbfloat[:, ::1]),
        ('penalized_objectives', nbfloat[:, ::1]),
        ('feasible_mask', boolean[:, ::1]),
        ('fitnesses', nbfloat[:, ::1]),
        ('noptimal_mask', boolean[:, ::1]),
        ('scaled_centroids', nbfloat[:, ::1]),
        ('parents', nbfloat[:, :, ::1]),
        ('parent_indices', nbintp[::1]),

        # Optima tracking
        ('optima_points', nbfloat[:, ::1]),
        ('optima_scaled_points', nbfloat[:, ::1]),
        ('optima_raw_objectives', nbfloat[::1]),
        ('optima_violations', nbfloat[::1]),
        ('optima_penalized_objectives', nbfloat[::1]),
        ('optima_fitnesses', nbfloat[::1]),
        ('optima_noptimal_mask', boolean[::1]),
        ('current_optimum', nbfloat),
        ('noptimal_threshold', nbfloat),

        # Hyperparameters
        # (Defined individually in spec as they are assigned in update_hyperparameters)
        ('champ_count', nbint),
        ('elite_count', nbint),
        ('tourn_count', nbint),
        ('tourn_size', nbint),
        ('mutation_prob', nbfloat[:]),
        ('mutation_sigma', nbfloat[:]),
        ('mutation_alpha', nbfloat),
        ('crossover_prob', nbfloat[:]),
        ('noptimal_rel', nbfloat),
        ('noptimal_abs', nbfloat),
        ('violation_factor', nbfloat),
        ('niche_elitism', boolean),
        ('mutation_scaler', nbfloat[::1]),
        ('space_scaler', nbfloat[:, ::1]),
        ('objective_scaler', nbfloat),
        ('repulsion_weight', nbfloat),
        ('fitness_method', nbint),

        ('scaled_ndim', nbint),
        ('log_mutation_sigma', nbfloat[:]),
        ('current_mutation_prob', nbfloat),
        ('current_mutation_sigma', nbfloat),
        ('current_crossover_prob', nbfloat),

        ('hyperparameters_set', boolean),
        ('do_skew_mutation', boolean),
        ('use_cx_two_point', boolean),

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
        include_obj_in_fitness: bool,
        pure_optimization: bool,
    ):
        self.num_niches = nbint(num_niches)
        self.pop_size = nbint(pop_size)
        self.ndim = nbint(ndim)
        self.parent_size = nbint(0)
        self.stable_sort = stable_sort
        self.rng = rng
        self.include_obj_in_fitness = include_obj_in_fitness
        self.pure_optimization = pure_optimization

        if self.pure_optimization:
            if self.include_obj_in_fitness:
                print("Warning: include_obj_in_fitness has no effect when pure_optimization is True.")
                self.include_obj_in_fitness = False

        self.integrality = np.empty(ndim, np.bool_)
        self.booleanality = np.empty(ndim, np.bool_)
        self.maximize = False
        self.scaling_in_obj_func = False
        self.lower_bounds = np.empty(ndim, dtype=npfloat)
        self.upper_bounds = np.empty(ndim, dtype=npfloat)
        self.problem_loaded = False
        self.hyperparameters_set = False
        self.optimum_given = False

        # Hyperparameters
        self.champ_count = 0
        self.elite_count = 0
        self.tourn_count = 0
        self.tourn_size = 0
        self.mutation_prob = np.empty(2, dtype=npfloat)
        self.mutation_sigma = np.empty(2, dtype=npfloat)
        self.log_mutation_sigma = np.empty(2, dtype=npfloat)
        self.mutation_alpha = 0.0
        self.crossover_prob = np.empty(2, dtype=npfloat)
        self.niche_elitism = False
        self.noptimal_rel = 0.0
        self.noptimal_abs = 0.0
        self.violation_factor = 0.0
        self.mutation_scaler = np.empty(ndim, dtype=npfloat)
        self.space_scaler = np.empty((1, ndim), dtype=npfloat)
        self.objective_scaler = 1.0
        self.repulsion_weight = 0.0
        self.fitness_method = 2

        self.scaled_ndim = nbint(ndim)
        self.do_skew_mutation = False
        self.use_cx_two_point = self.ndim > 2

        # Diversity loggers
        self.vesa = 0.0
        self.shannon = 0.0
        self.mean_fitness = 0.0
        self.stds = np.empty(0, dtype=npfloat)  # len=num_niches
        self.variances = np.empty(0, dtype=npfloat)  # len=num_niches

        # Population data arrays
        self.points = np.empty((self.num_niches, self.pop_size, self.ndim), dtype=npfloat)
        self.raw_objectives = np.empty((self.num_niches, self.pop_size), dtype=npfloat)
        self.violations = np.zeros((self.num_niches, self.pop_size), dtype=npfloat)
        self.penalized_objectives = np.empty((self.num_niches, self.pop_size), dtype=npfloat)
        self.feasible_mask = np.empty((self.num_niches, self.pop_size), dtype=np.bool_)
        self.fitnesses = np.empty((self.num_niches, self.pop_size), dtype=npfloat)
        self.noptimal_mask = np.empty((self.num_niches, self.pop_size), dtype=np.bool_)
        self.parents = np.empty((self.num_niches, 0, self.ndim), dtype=npfloat)
        self.parent_indices = np.empty((0,), dtype=npintp)

        # Overall best found
        self.optima_points = np.empty((self.num_niches, self.ndim), dtype=npfloat)
        self.optima_raw_objectives = np.empty((self.num_niches), dtype=npfloat)
        self.optima_violations = np.empty((self.num_niches), dtype=npfloat)
        self.optima_penalized_objectives = np.empty((self.num_niches), dtype=npfloat)
        self.optima_fitnesses = np.empty((self.num_niches), dtype=npfloat)
        self.optima_noptimal_mask = np.empty((self.num_niches), dtype=np.bool_)
        self.noptimal_threshold = 0.0
        self.current_optimum = 0.0

        self._allocate_scaled_arrays()

    def _load_optimum(
            self,
            point,  # np.NDArray[float]
            objective: float,
            violation: float,
            violation_factor: float,
    ):
        if not (point.ndim == 0 or point.size == 0):
            if not point.ndim == 1:
                raise ValueError("optimum point should be 1D")
            if not point.shape[0] == self.ndim:
                raise ValueError(f"optimum point should have length {self.ndim}. Got: {point.shape[0]}")

            self.optimum_given = True
            self.optima_points[0, :] = point

        self.optima_raw_objectives[0] = objective
        self.optima_violations[0] = violation
        self.optima_penalized_objectives[0] = objective + violation_factor * violation
        self.optima_noptimal_mask[0] = False  # will be updated later

    def initialize_population(  # noqa: C901
            self,
            points,  # : NDArray[float]
    ):
        """
        populates the population points with a uniform distribution
        """
        if not self.problem_loaded:
            raise RuntimeError("Problem not loaded. Load problem before populating.")
        if not self.hyperparameters_set:
            raise RuntimeError("Please set hyperparameters before populating")

        if points.ndim == 0 or points.size == 0:
            self._populate_randomly(0, self.num_niches)
        else:
            if points.ndim == 1:
                if not points.shape[0] == self.ndim:
                    raise ValueError(f"1D 'points' should have shape ({self.ndim},). Got {points.shape}.")

                parent_size = self.parent_size
                self._resize_parent_size(1)

                for i in range(self.num_niches):
                    self.parents[i, 0, :] = points
                self.generate_offspring()

                self._resize_parent_size(parent_size)

            elif points.ndim == 2:
                if not (points.shape[0] == self.pop_size
                        and points.shape[1] == self.ndim):
                    raise ValueError(
                        f"2D 'points' should have shape (pop_size={self.pop_size}, ndim={self.ndim}). Got {points}."
                        " If you are supplying (num_niches, ndim), try reshaping to (num_niches, 1, ndim)"
                    )

                for i in range(self.num_niches):
                    self.points[i, :, :] = points

            elif points.ndim == 3:
                if not (points.shape[0] == self.num_niches
                        and points.shape[1] == self.pop_size
                        and points.shape[2] == self.ndim):
                    raise ValueError(
                        f"3D 'points' should have shape (num_niches={self.num_niches}, pop_size={self.pop_size},"
                        f"ndim={self.ndim}). Got {points.shape}"
                    )
                _clone_3d(self.points, points)

            else:
                raise ValueError(f"'points' should be 1D, 2D, or 3D array. Got {points.ndim}D array.")

        if self.optimum_given:
            self.points[0, 0, :] = self.optima_points[0]  # inject known optimum

        self._apply_integrality()
        self._apply_bounds()
        self._scale_points()

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

        # TODO: strategy = clone, clone with recombination
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
        champ_count: int,
        elite_count: int,
        tourn_count: int,
        tourn_size: int,
        mutation_prob: np.ndarray[float],
        mutation_sigma: np.ndarray[float],
        mutation_alpha: float,
        crossover_prob: np.ndarray[float],
        niche_elitism: bool,
        noptimal_rel: float,
        noptimal_abs: float,
        violation_factor: float,
        mutation_scaler: np.ndarray[float],
        space_scaler: np.ndarray[float],
        objective_scaler: float,
        repulsion_weight: float,
        fitness_method: int,
    ):
        self.champ_count = nbint(champ_count)
        self.elite_count = nbint(elite_count)
        self.tourn_count = nbint(tourn_count)
        self.tourn_size = nbint(tourn_size)
        self.mutation_prob[:] = mutation_prob.astype(nbfloat)
        self.mutation_sigma[:] = mutation_sigma.astype(nbfloat)
        self.log_mutation_sigma[:] = np.log(mutation_sigma)
        self.mutation_alpha = nbfloat(mutation_alpha)
        self.crossover_prob[:] = crossover_prob.astype(nbfloat)
        self.niche_elitism = boolean(niche_elitism)
        self.noptimal_rel = nbfloat(noptimal_rel)
        self.noptimal_abs = nbfloat(noptimal_abs)
        self.violation_factor = nbfloat(violation_factor)
        self.mutation_scaler[:] = mutation_scaler.astype(nbfloat)
        self.objective_scaler = nbfloat(objective_scaler)
        self.repulsion_weight = nbfloat(repulsion_weight)
        self.fitness_method = fitness_method

        self.do_skew_mutation = abs(self.mutation_alpha) > 1e-3

        self.space_scaler = np.ascontiguousarray(space_scaler).astype(nbfloat)
        new_scaled_ndim = space_scaler.shape[1]
        if new_scaled_ndim != self.scaled_ndim:
            self.scaled_ndim = nbint(new_scaled_ndim)
            self._allocate_scaled_arrays()

        self._rescale_bounds()
        self.dither_probabilities()

        self.hyperparameters_set = True

    def select_parents(self):
        """
        Selects parents for the next generation.
        """
        # Niche 0 (optimization) selects on objective value
        selection.selection(
            self.parent_indices,
            self.penalized_objectives[0],
            self.maximize,
            self.champ_count,
            self.elite_count,
            self.tourn_count,
            self.tourn_size,
            self.pop_size,
            self.rng,
            self.stable_sort
        )
        self.parents[0] = self.points[0, self.parent_indices]

        # Other niches (diversity) select on fitness, with objective as fallback
        for i in range(1, self.num_niches):
            selection.selection_with_fallback(
                self.parent_indices,
                self.fitnesses[i],
                self.noptimal_mask[i],
                self.penalized_objectives[i],
                self.maximize,
                self.champ_count,
                self.elite_count,
                self.tourn_count,
                self.tourn_size,
                self.pop_size,
                self.rng,
                self.stable_sort,
            )
            self.parents[i] = self.points[i, self.parent_indices]

    def generate_offspring(self):
        """
        Generates offspring from parents via cloning, crossover, and mutation.
        """
        _clone_3d(self.points, self.parents)

        self._crossover()
        self._mutate()
        self._preserve_elites()

        self._apply_bounds()
        self._scale_points()

    def _crossover(self):
        if self.use_cx_two_point:
            crossover.crossover_population(
                points=self.points,
                crossover_prob=self.current_crossover_prob,
                cx_func=crossover.cx_two_point,
                rng=self.rng,
            )
        else:
            crossover.crossover_population(
                points=self.points,
                crossover_prob=self.current_crossover_prob,
                cx_func=crossover.cx_one_point,
                rng=self.rng,
            )

    def _mutate(self):
        sigma_vals = self.current_mutation_sigma * self.mutation_scaler
        if self.is_continuous_space and self.do_skew_mutation:
            mutation.mutate_skew_population_float(
                points=self.points,
                mutation_sigma=sigma_vals,
                mutation_prob=self.current_mutation_prob,
                mutation_alpha=self.mutation_alpha,
                rng=self.rng,
            )
        elif self.is_continuous_space and not self.do_skew_mutation:
            mutation.mutate_gaussian_population_float(
                points=self.points,
                mutation_sigma=sigma_vals,
                mutation_prob=self.current_mutation_prob,
                rng=self.rng,
            )
        elif not self.is_continuous_space and self.do_skew_mutation:
            mutation.mutate_skew_population_mixed(
                points=self.points,
                mutation_sigma=sigma_vals,
                mutation_prob=self.current_mutation_prob,
                mutation_alpha=self.mutation_alpha,
                rng=self.rng,
                integrality=self.integrality,
                booleanality=self.booleanality,
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

    def _preserve_elites(self):
        # Elitism: preserve best individuals
        self.points[0, 0, :] = self.optima_points[0]
        if self.niche_elitism:
            # This part could be enhanced, for now, we just preserve the best from each parent set
            for i in range(1, self.num_niches):
                if self.optima_noptimal_mask[i]:  # only preserve if noptimal
                    self.points[i, 0, :] = self.optima_points[i, :]
                else:  # otherwise duplicate the current optimum
                    self.points[i, 0, :] = self.optima_points[0, :]

        if self.pure_optimization:
            for i in range(1, self.num_niches):
                self.points[i, 0, :] = self.optima_points[0, :]

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
                gi, gj = utils.argmax_2d(self.penalized_objectives)
            else:
                gi, gj = utils.argmin_2d(self.penalized_objectives)

        if gi != -1:
            self.optima_points[0, :] = self.points[gi, gj]
            self.optima_scaled_points[0, :] = self.scaled_points[gi, gj]
            self.optima_raw_objectives[0] = self.raw_objectives[gi, gj]
            self.optima_violations[0] = self.violations[gi, gj]
            self.optima_penalized_objectives[0] = self.penalized_objectives[gi, gj]
            self.optima_fitnesses[0] = self.fitnesses[gi, gj]
            # logically must be true but self.noptimal_mask has not been calculated yet
            self.optima_noptimal_mask[0] = True

        self.current_optimum = self.optima_penalized_objectives[0]
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

        if self.pure_optimization:
            mean_abs_dif = 0.0
            for i in range(1, self.num_niches):
                diff = self.optima_penalized_objectives[i] - self.current_optimum
                mean_abs_dif += -diff if self.maximize else diff

            mean_abs_dif /= (self.num_niches - 1)
            rel_dif = utils.safe_divide_scalar(mean_abs_dif, abs(self.current_optimum), mean_abs_dif)

            target_weight = 0.005 / (rel_dif)
            print("mean_rel_dif:", mean_abs_dif, ". old weight:", self.repulsion_weight,
                  ". new weight:", (0.8 * self.repulsion_weight) + (0.2 * target_weight))

            self.repulsion_weight = (0.8 * self.repulsion_weight) + (0.2 * target_weight)
            # does putting optimum in offspring counteract this??

    def evaluate_fitness(self):
        """
        Calculates fitness for each individual based on distance to other niche centroids.
        """
        self._find_scaled_centroids()

        if self.pure_optimization:
            fitness.evaluate_fitness_pure_optimization(
                self.fitnesses,
                self.scaled_points,
                self.scaled_centroids,
                self.raw_objectives,
                self.repulsion_weight,
                self.fitness_method,
            )
            return
        if self.fitness_method == 0:
            fitness.evaluate_fitness_angular(
                self.fitnesses,
                self.scaled_points,
                self.scaled_centroids,
                self.raw_objectives,
                self.objective_scaler,
                self.include_obj_in_fitness,
            )
        else:
            fitness.evaluate_fitness_distance(
                self.fitnesses,
                self.scaled_points,
                self.scaled_centroids,
                self.raw_objectives,
                self.objective_scaler,
                self.fitness_method,
                self.include_obj_in_fitness,
            )

    def evaluate_diversity(self):
        """
        Calculates and returns a vector of diversity metrics for the current population state.
        """
        scaled_nopt_points = self.optima_scaled_points[self.optima_noptimal_mask]

        # VESA
        if scaled_nopt_points.shape[0] >= 2:
            self.vesa = diversity.volume_estimation_by_shadow_addition(
                scaled_nopt_points, np.ones(scaled_nopt_points.shape[0], dtype=np.bool_)
            )
        else:
            self.vesa = 0.0

        # Shannon Index
        if scaled_nopt_points.shape[0] >= 1:
            self.shannon = diversity.mean_of_shannon_of_projections(
                scaled_nopt_points,
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
        self.current_mutation_prob = utils.uniform_dither(self.mutation_prob, self.rng)
        self.current_mutation_sigma = utils.loguniform_dither(self.log_mutation_sigma, self.rng)
        self.current_crossover_prob = utils.uniform_dither(self.crossover_prob, self.rng)

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
            self.parents = _add_niche_to_array(self.parents, new_niche_size)

            self.optima_points = _add_niche_to_array(self.optima_points, new_niche_size)
            self.optima_scaled_points = _add_niche_to_array(self.optima_scaled_points, new_niche_size)
            self.optima_raw_objectives = _add_niche_to_array(self.optima_raw_objectives, new_niche_size)
            self.optima_violations = _add_niche_to_array(self.optima_violations, new_niche_size)
            self.optima_penalized_objectives = _add_niche_to_array(self.optima_penalized_objectives, new_niche_size)
            self.optima_fitnesses = _add_niche_to_array(self.optima_fitnesses, new_niche_size)
            self.optima_noptimal_mask = _add_niche_to_array(self.optima_noptimal_mask, new_niche_size)

            self.num_niches = new_niche_size

    def _resize_pop_size(self, new_pop_size: int):
        if new_pop_size == self.pop_size:
            return

        # Create new arrays with the target size
        new_points = np.empty((self.num_niches, new_pop_size, self.ndim))
        new_scaled_points = np.empty((self.num_niches, new_pop_size, self.scaled_ndim), dtype=npfloat)
        new_raw_objectives = np.empty((self.num_niches, new_pop_size), dtype=npfloat)
        new_violations = np.zeros((self.num_niches, new_pop_size), dtype=npfloat)
        new_penalized_objectives = np.empty((self.num_niches, new_pop_size), dtype=npfloat)
        new_feasible_mask = np.empty((self.num_niches, new_pop_size), dtype=np.bool_)
        new_fitnesses = np.empty((self.num_niches, new_pop_size), dtype=npfloat)
        new_noptimal_mask = np.empty((self.num_niches, new_pop_size), dtype=np.bool_)

        if new_pop_size > self.pop_size:
            _clone_3d(new_points, self.points)
            _clone_3d(new_scaled_points, self.scaled_points)
            _clone_2d(new_raw_objectives, self.raw_objectives)
            _clone_2d(new_violations, self.violations)
            _clone_2d(new_penalized_objectives, self.penalized_objectives)
            _clone_2d(new_feasible_mask, self.feasible_mask)
            _clone_2d(new_fitnesses, self.fitnesses)
            _clone_2d(new_noptimal_mask, self.noptimal_mask)

        else:  # new_pop_size < self.pop_size
            # Decrease size: select the best individuals to keep

            indices = np.empty(new_pop_size, npintp)

            # Other niches (diversity) are selected based on fitness
            for i in range(self.num_niches):
                if i == 0:
                    # Niche 0 (optimization) is selected based on objective value
                    selection.selection(
                        indices,
                        self.penalized_objectives[0],
                        self.maximize,
                        self.champ_count,
                        self.elite_count,
                        self.tourn_count,
                        self.tourn_size,
                        self.pop_size,
                        self.rng,
                        self.stable_sort,
                    )
                else:
                    selection.selection_with_fallback(
                        indices,
                        self.fitnesses[i],
                        self.noptimal_mask[i],
                        self.penalized_objectives[i],
                        self.maximize,
                        self.champ_count,
                        self.elite_count,
                        self.tourn_count,
                        self.tourn_size,
                        self.pop_size,
                        self.rng,
                        self.stable_sort,
                    )

                new_points[i] = self.points[i, indices]
                new_scaled_points[i] = self.scaled_points[i, indices]
                new_raw_objectives[i] = self.raw_objectives[i, indices]
                new_violations[i] = self.violations[i, indices]
                new_penalized_objectives[i] = self.penalized_objectives[i, indices]
                new_feasible_mask[i] = self.feasible_mask[i, indices]
                new_fitnesses[i] = self.fitnesses[i, indices]
                new_noptimal_mask[i] = self.noptimal_mask[i, indices]

        # Replace points array and re-initialize metric arrays
        self.points = new_points
        self.scaled_points = new_scaled_points
        self.raw_objectives = new_raw_objectives
        self.violations = new_violations
        self.penalized_objectives = new_penalized_objectives
        self.feasible_mask = new_feasible_mask
        self.fitnesses = new_fitnesses
        self.noptimal_mask = new_noptimal_mask

        self.pop_size = new_pop_size

    def _resize_parent_size(self, new_parent_size):
        if new_parent_size != self.parent_size:
            if self.parent_size > 0:
                new_parents = np.empty((self.num_niches, new_parent_size, self.ndim), dtype=npfloat)
                _clone_3d(new_parents, self.parents)
                self.parents = new_parents
                self.parent_size = nbint(new_parent_size)
            else:
                self.parents = np.empty((self.num_niches, new_parent_size, self.ndim), dtype=npfloat)
                self.parent_size = nbint(new_parent_size)
            self.parent_indices = np.empty(self.parent_size, npintp)

    def _allocate_scaled_arrays(self):
        self.scaled_points = np.empty((self.num_niches, self.pop_size, self.scaled_ndim))
        self.optima_scaled_points = np.empty((self.num_niches, self.scaled_ndim))

        if self.pure_optimization:
            self.scaled_centroids = np.empty((1, self.scaled_ndim), dtype=npfloat)
        elif self.include_obj_in_fitness:
            self.scaled_centroids = np.empty((self.num_niches, self.scaled_ndim + 1), dtype=npfloat)
        else:
            self.scaled_centroids = np.empty((self.num_niches, self.scaled_ndim), dtype=npfloat)

        self.scaled_lower_bounds = np.empty(self.scaled_ndim, dtype=npfloat)
        self.scaled_upper_bounds = np.empty(self.scaled_ndim, dtype=npfloat)

    def _find_scaled_centroids(self):
        """
        Calculates the geometric center (centroid) of each niche.
        """
        self.scaled_centroids[:, :] = 0.0

        for i in range(self.num_niches):
            for j in range(self.pop_size):
                for sk in range(self.scaled_ndim):
                    self.scaled_centroids[i, sk] += self.scaled_points[i, j, sk]

        if self.include_obj_in_fitness:
            for i in range(self.num_niches):
                for j in range(self.pop_size):
                    self.scaled_centroids[i, self.scaled_ndim] += self.raw_objectives[i, j]

            self.scaled_centroids[:, self.scaled_ndim] *= self.objective_scaler

        self.scaled_centroids /= self.pop_size

        if self.fitness_method == 0:
            self.scaled_centroids[0, :self.scaled_ndim] = self.optima_scaled_points[0]
            if self.include_obj_in_fitness:
                self.scaled_centroids[0, self.scaled_ndim] = self.optima_raw_objectives[0] * self.objective_scaler

    def _scale_points(self):
        """project points onto other space defined by space_scaler"""
        if self.scaling_in_obj_func:
            return

        if self.space_scaler.shape[0] == 1:
            for i in range(self.num_niches):
                for j in range(self.pop_size):
                    for k in range(self.ndim):
                        self.scaled_points[i, j, k] = self.points[i, j, k] * self.space_scaler[0, k]

        else:
            for i in range(self.num_niches):
                for j in range(self.pop_size):
                    for sk in range(self.scaled_ndim):
                        val = 0.0
                        for k in range(self.ndim):
                            val += self.points[i, j, k] * self.space_scaler[k, sk]
                        self.scaled_points[i, j, sk] = val

    def _rescale_bounds(self):
        if self.space_scaler.shape[0] == 1:
            for k in range(self.ndim):
                self.scaled_lower_bounds[k] = self.lower_bounds[k] * self.space_scaler[0, k]
                self.scaled_upper_bounds[k] = self.upper_bounds[k] * self.space_scaler[0, k]
        else:
            for sk in range(self.scaled_ndim):
                min_val = 0.0
                max_val = 0.0
                for k in range(self.ndim):
                    p = self.space_scaler[k, sk]
                    if p > 0:
                        min_val += self.lower_bounds[k] * p
                        max_val += self.upper_bounds[k] * p
                    else:
                        min_val += self.upper_bounds[k] * p
                        max_val += self.lower_bounds[k] * p
                self.scaled_lower_bounds[sk] = min_val
                self.scaled_upper_bounds[sk] = max_val

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
        margin = abs(self.current_optimum) * (self.noptimal_rel) + self.noptimal_abs

        if self.maximize:
            self.noptimal_threshold = self.current_optimum - margin
        else:
            self.noptimal_threshold = self.current_optimum + margin

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
            best_i, best_j = utils.argmax_with_mask_2d(
                self.penalized_objectives, self.feasible_mask, self.optima_penalized_objectives[0]
            )
        else:
            best_i, best_j = utils.argmin_with_mask_2d(
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
                best_j = utils.argmax_with_mask(self.fitnesses[i], self.feasible_mask[i])

            else:
                # option 2: best penalized objective
                best_j = utils.argm(self.penalized_objectives[i], self.maximize)

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
def _clone_3d(target, parents):
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
def _clone_2d(target, parents):
    nniches = parents.shape[0]
    nindividuals = parents.shape[1]
    for i in range(target.shape[0]):
        i_n = i % nniches
        for j in range(target.shape[1]):
            j_n = j % nindividuals
            target[i, j] = parents[i_n, j_n]


@njit
def njit_deepcopy(new, old):
    flat_new = new.ravel()
    flat_old = old.ravel()

    for i in range(flat_old.shape[0]):
        flat_new[i] = flat_old[i]


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
