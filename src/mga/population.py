import numpy as np
from numba.types import npy_rng

from mga.commons.constants import JIT_ENABLED
from mga.commons.numba_overload import njit, jitclass
from mga.commons.types import nbint, nbintp, npintp, nbfloat, npfloat, boolean, npint
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
        ('scaled_ndim', nbint),
        ('parent_size', nbint),
        ('stable_sort', boolean),
        ('include_obj_in_fitness', boolean),
        ('pure_optimization', boolean),

        ('pure_optimization', boolean),
        ('niche_types', nbint[::1]),
        ('niche_target_dims', nbint[::1]),
        ('niche_is_min', boolean[::1]),
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
        ('extrema_true_points', nbfloat[:, ::1]),
        ('extrema_true_vals', nbfloat[::1]),
        ('extrema_true_penalized_objectives', nbfloat[::1]),
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
        ('extrema_epsilon_rel', nbfloat),
        ('extrema_epsilon_abs', nbfloat),

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
        niche_types: np.ndarray,
        niche_target_dims: np.ndarray,
        niche_is_min: np.ndarray,
    ):
        self.num_niches = nbint(num_niches)
        self.pop_size = nbint(pop_size)
        self.ndim = nbint(ndim)
        self.scaled_ndim = nbint(ndim)
        self.parent_size = nbint(0)
        self.stable_sort = stable_sort
        self.rng = rng
        self.include_obj_in_fitness = include_obj_in_fitness
        self.pure_optimization = pure_optimization

        if self.pure_optimization:
            if self.include_obj_in_fitness:
                print("Warning: include_obj_in_fitness has no effect when pure_optimization is True.")
                self.include_obj_in_fitness = False

        self.niche_types = niche_types
        self.niche_target_dims = niche_target_dims
        self.niche_is_min = niche_is_min
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
        self.extrema_epsilon_rel = 1e-3
        self.extrema_epsilon_abs = 1e-6

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
        self.points = np.zeros((self.num_niches, self.pop_size, self.ndim), dtype=npfloat)
        self.raw_objectives = np.zeros((self.num_niches, self.pop_size), dtype=npfloat)
        self.violations = np.zeros((self.num_niches, self.pop_size), dtype=npfloat)
        self.penalized_objectives = np.zeros((self.num_niches, self.pop_size), dtype=npfloat)
        self.feasible_mask = np.zeros((self.num_niches, self.pop_size), dtype=np.bool_)
        self.fitnesses = np.zeros((self.num_niches, self.pop_size), dtype=npfloat)
        self.noptimal_mask = np.zeros((self.num_niches, self.pop_size), dtype=np.bool_)
        self.parents = np.zeros((self.num_niches, 0, self.ndim), dtype=npfloat)
        self.parent_indices = np.zeros((0,), dtype=npintp)

        # Overall best found
        self.optima_points = np.zeros((self.num_niches, self.ndim), dtype=npfloat)
        self.optima_raw_objectives = np.zeros((self.num_niches), dtype=npfloat)
        self.optima_violations = np.zeros((self.num_niches), dtype=npfloat)
        self.optima_penalized_objectives = np.zeros((self.num_niches), dtype=npfloat)
        self.optima_fitnesses = np.zeros((self.num_niches), dtype=npfloat)
        self.optima_noptimal_mask = np.zeros((self.num_niches), dtype=np.bool_)
        self.extrema_true_points = np.zeros((self.num_niches, self.ndim), dtype=npfloat)
        self.extrema_true_vals = np.zeros((self.num_niches), dtype=npfloat)
        self.extrema_true_penalized_objectives = np.full(
            (self.num_niches), -np.inf if self.maximize else np.inf, dtype=npfloat
        )
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

    def _populate_from_noptimal_pool(self, start_idx: int, end_idx: int):
        """
        Seeds newly added niches (`start_idx` to `end_idx`) from the existing pool of
        n-optimal champions and feasible n-optimal population points:
        - `points[i, 0]` receives a distinct existing n-optimal champion (cycling through
          extrema and diversity champions).
        - `points[i, 1:]` are generated via uniform crossover + Gaussian mutation of random
          pairs drawn from the global feasible n-optimal pool.
        """
        # 1. Gather distinct n-optimal champions from existing niches [0, start_idx)
        champ_indices = np.empty(start_idx, dtype=npintp)
        n_champs = 0
        for c in range(start_idx):
            if self.optima_noptimal_mask[c]:
                is_dup = False
                for prev in range(n_champs):
                    if selection.is_duplicate_point(
                        self.optima_scaled_points[c],
                        self.optima_scaled_points[champ_indices[prev]],
                    ):
                        is_dup = True
                        break
                if not is_dup:
                    champ_indices[n_champs] = c
                    n_champs += 1

        # 2. Gather flat indices of all feasible & n-optimal individuals in [0, start_idx)
        max_pool = start_idx * self.pop_size
        pool_niche = np.empty(max_pool, dtype=npintp)
        pool_ind = np.empty(max_pool, dtype=npintp)
        n_pool = 0
        for gi in range(start_idx):
            for gj in range(self.pop_size):
                if self.feasible_mask[gi, gj]:
                    pool_niche[n_pool] = gi
                    pool_ind[n_pool] = gj
                    n_pool += 1

        if n_champs == 0 and n_pool == 0:
            self._populate_randomly(start_idx, end_idx)
            return

        sigma_vals = self.current_mutation_sigma * self.mutation_scaler
        mut_prob = self.current_mutation_prob

        for i in range(start_idx, end_idx):
            # Seed slot 0 with a distinct n-optimal champion (preferring non-zero niches first)
            if n_champs > 0:
                # Offset by 1 if n_champs > 1 so new niches start from extrema/diversity corners before niche 0
                c_idx = champ_indices[(i - start_idx + (1 if n_champs > 1 else 0)) % n_champs]
                self.points[i, 0, :] = self.optima_points[c_idx, :]
                self.optima_points[i, :] = self.optima_points[c_idx, :]
                self.optima_scaled_points[i, :] = self.optima_scaled_points[c_idx, :]
                self.optima_raw_objectives[i] = self.optima_raw_objectives[c_idx]
                self.optima_violations[i] = self.optima_violations[c_idx]
                self.optima_penalized_objectives[i] = self.optima_penalized_objectives[c_idx]
                self.optima_noptimal_mask[i] = True
            else:
                p_idx = (i - start_idx) % n_pool
                gi, gj = pool_niche[p_idx], pool_ind[p_idx]
                self.points[i, 0, :] = self.points[gi, gj, :]
                self._set_optimum(i, gi, gj)

            # Seed slots 1..pop_size-1 by recombining and mutating pairs from the feasible n-optimal pool
            for j in range(1, self.pop_size):
                if n_pool >= 2:
                    idx1 = self.rng.integers(0, n_pool)
                    idx2 = self.rng.integers(0, n_pool)
                    p1 = self.points[pool_niche[idx1], pool_ind[idx1]]
                    p2 = self.points[pool_niche[idx2], pool_ind[idx2]]
                elif n_champs >= 2:
                    idx1 = self.rng.integers(0, n_champs)
                    idx2 = self.rng.integers(0, n_champs)
                    p1 = self.optima_points[champ_indices[idx1]]
                    p2 = self.optima_points[champ_indices[idx2]]
                else:
                    p1 = self.points[i, 0]
                    p2 = self.points[i, 0]

                for k in range(self.ndim):
                    val = p1[k] if self.rng.random() < 0.5 else p2[k]
                    if self.rng.random() < mut_prob:
                        val += self.rng.normal(0.0, 1.0) * sigma_vals[k]
                    self.points[i, j, k] = val

    def add_diversity_niches(self, num_new_niches: int):
        if num_new_niches <= 0:
            return
        num_old = self.num_niches
        num_total = num_old + num_new_niches

        self.niche_types = _append_1d_int(self.niche_types, np.full(num_new_niches, 2, dtype=npint))
        self.niche_target_dims = _append_1d_int(self.niche_target_dims, np.full(num_new_niches, -1, dtype=npint))
        self.niche_is_min = _append_1d_bool(self.niche_is_min, np.zeros(num_new_niches, dtype=np.bool_))

        self.resize(num_niches=num_total)
        self._add_niche_clean_up(num_old, num_total)

    def add_extrema_niches(self, target_dims: np.ndarray[int], is_min: np.ndarray[bool]):
        num_old = self.num_niches
        num_new_niches = len(target_dims)
        num_total = num_old + num_new_niches

        self.niche_types = _append_1d_int(self.niche_types, np.full(num_new_niches, 1, dtype=npint))
        self.niche_target_dims = _append_1d_int(self.niche_target_dims, target_dims)
        self.niche_is_min = _append_1d_bool(self.niche_is_min, is_min)

        self.resize(num_niches=num_total)
        self._add_niche_clean_up(num_old, num_total)

    def _add_niche_clean_up(self, num_old: int, num_total: int):
        if (
            self.problem_loaded and self.hyperparameters_set
            and num_old > 0 and np.any(self.optima_noptimal_mask[:num_old])
        ):
            self._populate_from_noptimal_pool(num_old, num_total)
        else:
            self._populate_randomly(num_old, num_total)
        self._apply_integrality()
        self._apply_bounds()
        self._scale_points()

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
        extrema_epsilon_rel: float,
        extrema_epsilon_abs: float,
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
        self.extrema_epsilon_rel = nbfloat(extrema_epsilon_rel)
        self.extrema_epsilon_abs = nbfloat(extrema_epsilon_abs)

        self.do_skew_mutation = abs(self.mutation_alpha) > 1e-3

        self.space_scaler = np.ascontiguousarray(space_scaler).astype(nbfloat)
        if not self.scaling_in_obj_func:
            new_scaled_ndim = space_scaler.shape[1]
            if new_scaled_ndim != self.scaled_ndim:
                self.scaled_ndim = nbint(new_scaled_ndim)
                self._allocate_scaled_arrays()
            self._rescale_bounds()

        self.dither_probabilities()

        self.hyperparameters_set = True

    def _generate_selection_indices(
        self,
        i: int,
        out_indices: np.ndarray
    ):
        """Generates selection indices based on niche type and selection pressure."""
        if i == 0:
            selection.selection(
                out_indices,
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
        else:
            if self.niche_types[i] == 1:  # Extrema niche
                dim_k = self.niche_target_dims[i]
                sec_obj = self.scaled_points[i, :, dim_k]
                maximize_sec = not self.niche_is_min[i]
                eps_rel = self.extrema_epsilon_rel
                eps_abs = self.extrema_epsilon_abs
            else:  # Diversity niche (type 2)
                sec_obj = self.fitnesses[i]
                maximize_sec = True
                eps_rel = 0.0
                eps_abs = 0.0

            selection.selection_with_fallback(
                out_indices,
                sec_obj,
                self.noptimal_mask[i],
                self.penalized_objectives[i],
                self.maximize,
                maximize_sec,
                eps_rel,
                eps_abs,
                self.champ_count,
                self.elite_count,
                self.tourn_count,
                self.tourn_size,
                self.pop_size,
                self.rng,
                self.stable_sort,
            )

    def select_parents(self):
        """
        Selects parents for the next generation:
        - Niche 0 (cost optimum): global champ + global dedup'ed elites across all niches, local tournament.
        - Extrema niches (type 1): global champ + global dedup'ed elites on `dim_k` across all niches, local tournament.
        - Diversity niches (type 2): local niche selection.
        """
        total_pop = self.num_niches * self.pop_size
        flat_points = self.points.reshape((total_pop, self.ndim))
        flat_scaled_points = self.scaled_points.reshape((total_pop, self.scaled_ndim))
        flat_pen_obj = self.penalized_objectives.reshape(total_pop)
        flat_nopt_mask = self.noptimal_mask.reshape(total_pop)
        # Note: track_optima() sets self.feasible_mask *= self.noptimal_mask - check violations == 0 for raw feasibility
        flat_feas_mask = (self.violations == 0.0).reshape(total_pop)

        for i in range(self.num_niches):
            local_offset = i * self.pop_size

            if i == 0:
                selection.selection_global_cost(
                    self.parent_indices,
                    flat_pen_obj,
                    flat_scaled_points,
                    local_offset,
                    self.pop_size,
                    self.maximize,
                    self.champ_count,
                    self.elite_count,
                    self.tourn_count,
                    self.tourn_size,
                    self.rng,
                    self.stable_sort,
                )
                self.parents[0] = flat_points[self.parent_indices]

            elif self.niche_types[i] == 1:
                dim_k = self.niche_target_dims[i]
                flat_dim_vals = np.ascontiguousarray(self.scaled_points[:, :, dim_k]).reshape(total_pop)
                maximize_sec = not self.niche_is_min[i]

                selection.selection_global_extrema(
                    self.parent_indices,
                    flat_dim_vals,
                    flat_nopt_mask,
                    flat_feas_mask,
                    flat_pen_obj,
                    flat_scaled_points,
                    self.noptimal_threshold,
                    local_offset,
                    self.pop_size,
                    self.maximize,
                    maximize_sec,
                    self.extrema_epsilon_rel,
                    self.extrema_epsilon_abs,
                    self.champ_count,
                    self.elite_count,
                    self.tourn_count,
                    self.tourn_size,
                    self.rng,
                    self.stable_sort,
                )
                self.parents[i] = flat_points[self.parent_indices]

            else:
                self._generate_selection_indices(i, self.parent_indices)
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
            for i in range(1, self.num_niches):
                if self.optima_noptimal_mask[i]:  # only preserve if noptimal
                    self.points[i, 0, :] = self.optima_points[i, :]
                    if self.niche_types[i] == 1 and self.pop_size > 1:
                        self.points[i, 1, :] = self.extrema_true_points[i, :]
                else:  # otherwise duplicate the current optimum
                    self.points[i, 0, :] = self.optima_points[0, :]

        if self.pure_optimization:
            for i in range(1, self.num_niches):
                self.points[i, 0, :] = self.optima_points[0, :]

    def _set_optimum(
        self,
        target_niche: int,
        source_niche: int,
        source_ind: int
    ):
        """Helper to copy an individual's properties into the optima history arrays."""
        self.optima_points[target_niche, :] = self.points[source_niche, source_ind, :]
        self.optima_scaled_points[target_niche, :] = self.scaled_points[source_niche, source_ind, :]
        self.optima_raw_objectives[target_niche] = self.raw_objectives[source_niche, source_ind]
        self.optima_violations[target_niche] = self.violations[source_niche, source_ind]
        self.optima_penalized_objectives[target_niche] = self.penalized_objectives[source_niche, source_ind]
        self.optima_fitnesses[target_niche] = self.fitnesses[source_niche, source_ind]
        self.optima_noptimal_mask[target_niche] = self.noptimal_mask[source_niche, source_ind]

    def track_optima(self):
        self._update_feasible_mask()
        self._track_global_optimum()
        self._update_noptimal_threshold()
        self._update_noptimal_mask()
        self.feasible_mask *= self.noptimal_mask

        self._track_extrema_niches()
        self._track_diversity_niches()

        if self.pure_optimization:
            self._track_pure_optimization()

    def _track_global_optimum(self):
        if np.any(self.feasible_mask):
            gi, gj = self._find_global_best_idx()
        else:
            if self.maximize:
                gi, gj = utils.argmax_2d(self.penalized_objectives)
            else:
                gi, gj = utils.argmin_2d(self.penalized_objectives)

        if gi != -1:
            self._set_optimum(0, gi, gj)
            self.optima_noptimal_mask[0] = True

        self.current_optimum = self.optima_penalized_objectives[0]

    def _find_true_extremum(self, dim_k, is_min, has_true, true_val, true_obj):
        """Scans the population for the strict true extremum among n-optimal points."""
        true_gi, true_gj = nbint(-1), nbint(-1)
        for gi in range(self.num_niches):
            for gj in range(self.pop_size):
                if self.feasible_mask[gi, gj] and self.noptimal_mask[gi, gj]:
                    val = self.scaled_points[gi, gj, dim_k]
                    obj = self.penalized_objectives[gi, gj]
                    if not has_true:
                        true_val, true_obj, true_gi, true_gj = val, obj, gi, gj
                        has_true = True
                    elif (is_min and val < true_val) or (not is_min and val > true_val):
                        true_val, true_obj, true_gi, true_gj = val, obj, gi, gj
                    elif val == true_val:
                        if (self.maximize and obj > true_obj) or (not self.maximize and obj < true_obj):
                            true_obj, true_gi, true_gj = obj, gi, gj
        return has_true, true_val, true_obj, true_gi, true_gj

    def _find_relaxed_champion(self, dim_k, is_min, true_val, eff_epsilon, has_champ, best_obj, best_val):
        """Finds the lowest-cost champion within eff_epsilon of the true extremum."""
        champ_gi, champ_gj = nbint(-1), nbint(-1)
        for gi in range(self.num_niches):
            for gj in range(self.pop_size):
                if self.feasible_mask[gi, gj] and self.noptimal_mask[gi, gj]:
                    val = self.scaled_points[gi, gj, dim_k]
                    dist = (val - true_val) if is_min else (true_val - val)
                    if dist <= eff_epsilon:
                        obj = self.penalized_objectives[gi, gj]
                        if not has_champ:
                            best_obj, best_val, champ_gi, champ_gj = obj, val, gi, gj
                            has_champ = True
                        else:
                            obj_diff = (obj - best_obj) if self.maximize else (best_obj - obj)
                            if obj_diff > 0.0:
                                best_obj, best_val, champ_gi, champ_gj = obj, val, gi, gj
                            elif obj_diff == 0.0:
                                val_diff = (best_val - val) if is_min else (val - best_val)
                                if val_diff > 0.0:
                                    best_val, champ_gi, champ_gj = val, gi, gj
        return champ_gi, champ_gj

    def _track_extrema_niches(self):
        for i in range(1, self.num_niches):
            if self.niche_types[i] != 1:
                continue

            dim_k = self.niche_target_dims[i]
            is_min = self.niche_is_min[i]

            champ_nopt, true_nopt = False, False
            if self.optima_noptimal_mask[i] and self.optima_violations[i] == 0.0:
                if self.maximize:
                    champ_nopt = self.optima_penalized_objectives[i] >= self.noptimal_threshold
                    true_nopt = self.extrema_true_penalized_objectives[i] >= self.noptimal_threshold
                else:
                    champ_nopt = self.optima_penalized_objectives[i] <= self.noptimal_threshold
                    true_nopt = self.extrema_true_penalized_objectives[i] <= self.noptimal_threshold

            true_val = self.extrema_true_vals[i] if true_nopt else 0.0
            true_obj = (
                self.extrema_true_penalized_objectives[i]
                if true_nopt
                else (-np.inf if self.maximize else np.inf)
            )

            has_true, true_val, true_obj, true_gi, true_gj = self._find_true_extremum(
                dim_k, is_min, true_nopt, true_val, true_obj
            )

            if not has_true:
                best_j = utils.argm(self.penalized_objectives[i], self.maximize)
                self._set_optimum(i, i, best_j)
                self.extrema_true_points[i, :] = self.points[i, best_j, :]
                self.extrema_true_vals[i] = self.scaled_points[i, best_j, dim_k]
                self.extrema_true_penalized_objectives[i] = self.penalized_objectives[i, best_j]
                self.optima_noptimal_mask[i] = False
                continue

            self.extrema_true_vals[i] = true_val
            self.extrema_true_penalized_objectives[i] = true_obj
            if true_gi != -1:
                self.extrema_true_points[i, :] = self.points[true_gi, true_gj, :]

            eff_epsilon = abs(true_val) * self.extrema_epsilon_rel + self.extrema_epsilon_abs
            best_obj = -np.inf if self.maximize else np.inf
            best_val = true_val
            has_champ = False

            if champ_nopt:
                inc_val = self.optima_scaled_points[i, dim_k]
                if ((inc_val - true_val) if is_min else (true_val - inc_val)) <= eff_epsilon:
                    has_champ, best_obj, best_val = True, self.optima_penalized_objectives[i], inc_val

            champ_gi, champ_gj = self._find_relaxed_champion(
                dim_k, is_min, true_val, eff_epsilon, has_champ, best_obj, best_val
            )

            # Ensure the mask correctly invalidates if the old champ fell out of n-optimality
            # and no new champion was found in the epsilon band
            self.optima_noptimal_mask[i] = champ_nopt

            if champ_gi != -1:
                self._set_optimum(i, champ_gi, champ_gj)
                self.optima_noptimal_mask[i] = True

    def _track_diversity_niches(self):
        self._find_best_in_niche(1)

    def _track_pure_optimization(self):
        mean_abs_dif = 0.0
        for i in range(1, self.num_niches):
            diff = self.optima_penalized_objectives[i] - self.current_optimum
            mean_abs_dif += -diff if self.maximize else diff

        mean_abs_dif /= (self.num_niches - 1)
        rel_dif = utils.safe_divide_scalar(mean_abs_dif, abs(self.current_optimum), mean_abs_dif)

        target_weight = 0.005 / (rel_dif)
        self.repulsion_weight = (0.8 * self.repulsion_weight) + (0.2 * target_weight)

    def evaluate_fitness(self):
        """
        Calculates fitness for each individual.
        Skips distance/angular evaluation if all non-optimum niches are extrema niches.
        """
        if self.pure_optimization:
            self._evaluate_pure_optimization_fitness()
            return

        if self.num_niches > 1:
            self._evaluate_diversity_fitness()

    def _evaluate_pure_optimization_fitness(self):
        self._find_scaled_centroids()
        fitness.evaluate_fitness_pure_optimization(
            self.fitnesses,
            self.scaled_points,
            self.scaled_centroids,
            self.raw_objectives,
            self.repulsion_weight,
            self.fitness_method,
        )

    def _evaluate_diversity_fitness(self):
        self._find_scaled_centroids()
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
            if self.scaling_in_obj_func:
                for sk in range(self.scaled_ndim):
                    self.scaled_lower_bounds[sk] = np.min(self.scaled_points[:, :, sk])
                    self.scaled_upper_bounds[sk] = np.max(self.scaled_points[:, :, sk])
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
            self.extrema_true_points = _add_niche_to_array(self.extrema_true_points, new_niche_size)
            self.extrema_true_vals = _add_niche_to_array(self.extrema_true_vals, new_niche_size)
            self.extrema_true_penalized_objectives = _add_niche_to_array(
                self.extrema_true_penalized_objectives, new_niche_size
            )

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

            for i in range(self.num_niches):
                self._generate_selection_indices(i, indices)

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
        self.scaled_points = np.zeros((self.num_niches, self.pop_size, self.scaled_ndim))
        self.optima_scaled_points = np.zeros((self.num_niches, self.scaled_ndim))

        if self.pure_optimization:
            self.scaled_centroids = np.zeros((1, self.scaled_ndim), dtype=npfloat)
        elif self.include_obj_in_fitness:
            self.scaled_centroids = np.zeros((self.num_niches, self.scaled_ndim + 1), dtype=npfloat)
        else:
            self.scaled_centroids = np.zeros((self.num_niches, self.scaled_ndim), dtype=npfloat)

        self.scaled_lower_bounds = np.zeros(self.scaled_ndim, dtype=npfloat)
        self.scaled_upper_bounds = np.zeros(self.scaled_ndim, dtype=npfloat)

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
        """
        if self.maximize:
            for i in range(self.num_niches):
                for j in range(self.pop_size):
                    self.noptimal_mask[i, j] = self.penalized_objectives[i, j] >= self.noptimal_threshold
        else:
            for i in range(self.num_niches):
                for j in range(self.pop_size):
                    self.noptimal_mask[i, j] = self.penalized_objectives[i, j] <= self.noptimal_threshold

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

    def _find_best_in_niche(self, start_idx: int = 1):
        """
        For each diversity niche i >= start_idx, pick the representative index:
        1) best fitness among feasible, noptimal, and non-duplicate points
           (checked against global optimum and all preceding n-optimal niches)
        2) fallback to best feasible & noptimal (if all are duplicates)
        3) else best penalized objective
        """
        out = np.full(self.num_niches, -1, dtype=npintp)
        candidate_mask = np.empty(self.pop_size, dtype=np.bool_)

        for i in range(start_idx, self.num_niches):
            if self.niche_types[i] != 2:
                continue

            n_valid = 0
            for j in range(self.pop_size):
                if not self.feasible_mask[i, j]:
                    candidate_mask[j] = False
                    continue

                # Check against global optimum (c=0) and already-assigned n-optimal champions (c < i)
                is_dup = selection.is_duplicate_point(
                    self.scaled_points[i, j], self.optima_scaled_points[0]
                )
                if not is_dup:
                    for c in range(1, i):
                        if self.optima_noptimal_mask[c]:
                            if selection.is_duplicate_point(
                                self.scaled_points[i, j], self.optima_scaled_points[c]
                            ):
                                is_dup = True
                                break

                candidate_mask[j] = not is_dup
                if not is_dup:
                    n_valid += 1

            if n_valid >= 1:
                best_j = utils.argmax_with_mask(self.fitnesses[i], candidate_mask)
            elif self.feasible_mask[i].sum() >= 1:
                best_j = utils.argmax_with_mask(self.fitnesses[i], self.feasible_mask[i])
            else:
                best_j = utils.argm(self.penalized_objectives[i], self.maximize)

            out[i] = best_j
            self._set_optimum(i, i, best_j)

        return out


# %%
@njit
def _add_niche_to_array(old_array, num_niches):
    if num_niches < old_array.shape[0]:
        raise Exception
    # 3. Dynamically create the new shape
    new_shape = (num_niches,) + old_array.shape[1:]
    # 4. Create the new array
    new_array = np.zeros(new_shape, dtype=old_array.dtype)
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


@njit
def _append_1d_int(arr, vals):
    res = np.empty(arr.size + vals.size, dtype=arr.dtype)
    res[:arr.size] = arr
    res[arr.size:] = vals
    return res


@njit
def _append_1d_bool(arr, vals):
    res = np.empty(arr.size + vals.size, dtype=arr.dtype)
    res[:arr.size] = arr
    res[arr.size:] = vals
    return res


@njit(inline="always")
def _is_duplicate_point(p1, p2, rtol=1e-5, atol=1e-6):
    """
    Fast early-exit check for whether two points (1D arrays) are identical
    within floating-point / near-clone tolerance.
    """
    for k in range(p1.shape[0]):
        diff = abs(p1[k] - p2[k])
        scale = max(abs(p1[k]), abs(p2[k]))
        if diff > atol + rtol * scale:
            return False
    return True


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

    if problem.return_scaled and problem.scaled_ndim != population.scaled_ndim:
        population.scaled_ndim = nbint(problem.scaled_ndim)
        population._allocate_scaled_arrays()

    population.is_continuous_space = (population.integrality.sum() == 0)
    population.problem_loaded = True
