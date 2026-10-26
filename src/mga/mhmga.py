import numpy as np
from datetime import datetime as dt
import warnings
from numba.core.registry import CPUDispatcher
from typing import Callable

from mga.commons.types import npint, npfloat
from mga.commons.numba_overload import njit
import mga.utils.termination as term
from mga.utils import typing
from mga.problem_definition import OptimizationProblem
from mga.population import Population, load_problem_to_population
from mga.utils.logger import Logger
from mga.evaluator import construct


_SENTINEL = object()


class MGAProblem:
    """
    Orchestrates the modelling to generate alternatives algorithm.
    Manages the optimization loop, problem state, and logging.
    """

    _DEFAULT_HYPERPARAMS = {
        "max_iter": 100,
        "pop_size": 100,
        "champ_count": 1,
        "elite_count": 0.2,
        "tourn_count": -1,
        "tourn_size": 2,
        "mutation_prob": 0.3,
        "mutation_sigma": 0.05,
        "mutation_alpha": 0.0,
        "crossover_prob": 0.4,
        "violation_factor": 1.0,
        "noptimal_rel": 0.0,
        "noptimal_abs": 0.0,
        "niche_elitism": True,
        "mutation_scaler": "bounds",
        "objective_scaler": None,
        "repulsion_weight": 1.0,
        "fitness_method": "l2",
        "extrema_epsilon_rel": 1e-3,
        "extrema_epsilon_abs": 1e-6,
    }

    def __init__(
        self,
        problem: OptimizationProblem,
        x0: np.ndarray[float] = None,
        starting_points: np.ndarray[float] = None,
        starting_objectives: np.ndarray[float] = None,
        starting_constraints: np.ndarray[float] = None,
        log_dir: str | None = None,
        log_freq: int = -1,
        random_seed: int | None = None,
        parallelize: bool = True,
        callback: Callable = None,
        include_obj_in_fitness: bool = False,
        pure_optimization: bool = False,
        space_scaler: np.ndarray | str | None = None,
    ):
        """
        Initializes the MGA algorithm
        """
        # sanitize inputs
        typing.sanitize_type(problem, OptimizationProblem, "problem")
        typing.sanitize_type(log_dir, (str, "none"), "log_dir")
        typing.sanitize_type(log_freq, "integer", "log_freq")
        typing.sanitize_range(log_freq, "log_freq", ge=-1)
        typing.sanitize_type(random_seed, ("integer", "none"), "random_seed")
        typing.sanitize_type(parallelize, "boolean", "parallelize")
        typing.sanitize_type(include_obj_in_fitness, "boolean", "include_obj_in_fitness")
        typing.sanitize_type(pure_optimization, "boolean", "pure_optimization")

        # Instantiation
        self.problem = problem
        self.rng = np.random.default_rng(random_seed)
        self.stable_sort = random_seed is not None
        self.logger = Logger(log_dir, log_freq, create_dir=True, ndim=self.problem.ndim) if log_dir else None
        self.pure_optimization = pure_optimization

        # Niche Configuration Recipe
        self.niche_types = [0]          # 0: optimum, 1: extrema, 2: diversity
        self.niche_target_dims = [-1]   # -1: None, >=0: scaled_dim index
        self.niche_is_min = [False]
        self.num_niches = 1

        if x0 is None:
            self.x0 = np.empty([], dtype=npfloat)
        else:
            typing.sanitize_array_type(x0, 'float', "x0")
            self.x0 = x0.astype(npfloat)

        if starting_points is None:
            self.starting_points = np.empty([], dtype=npfloat)
        else:
            self.starting_points = self._sanitize_seed_points(starting_points, "starting_points")
        self.starting_objectives = (
            None if starting_objectives is None else np.asarray(starting_objectives, dtype=npfloat)
        )
        self.starting_constraints = (
            None if starting_constraints is None else np.asarray(starting_constraints, dtype=npfloat)
        )
        self.population = None
        self.current_iter = 0
        self.start_time = dt.now()

        self.fitness_method_dict = {
            "angular": npint(0),
            "l1": npint(1),
            "l2": npint(2),
        }
        self.verbosity = 0
        self.disp_rate = 0

        # Base State Configuration
        self._is_populated = False
        self.current_best_obj = npfloat(0)
        self.mean_fitness = npfloat(0)
        self.callback = callback
        self.include_obj_in_fitness = include_obj_in_fitness
        self.hyperparameters_set = False
        self._stop_requested = False

        # Hyperparameter Defaults
        self.reset_hyperparameters()
        self.space_scaler = None
        if space_scaler is not None:
            self.update_hyperparameters(space_scaler=space_scaler, mark_as_set=False)

        # Evaluation Paths
        self._can_use_fast_path = self.problem.objective_jitted and not self.problem.fkwargs

        self.evaluator = construct(
            self._can_use_fast_path,
            self.problem.vectorized,
            self.problem.constraints,
            self.problem.return_scaled,
            parallelize,  # ignored if _can_use_fast_path is False
        )

    @property
    def scaled_ndim(self) -> int:
        """Resolves current scaled_ndim from problem or space_scaler."""
        if self.problem.return_scaled:
            return int(self.problem.scaled_ndim)
        if hasattr(self, "space_scaler") and self.space_scaler is not None:
            if self.space_scaler.ndim == 2 and self.space_scaler.shape[0] != 1:
                return int(self.space_scaler.shape[1])
        return int(self.problem.ndim)

    def add_diversity_niches(self, num_niches: int):
        """Adds standard centroid-repulsion diversity niches."""
        typing.sanitize_type(num_niches, "integer", "num_niches")
        typing.sanitize_range(num_niches, "num_niches", ge=1)

        self.niche_types.extend([2] * num_niches)
        self.niche_target_dims.extend([-1] * num_niches)
        self.niche_is_min.extend([False] * num_niches)
        self.num_niches += num_niches

        if self._is_populated:
            self.population.add_diversity_niches(num_niches)

    def add_extrema_niches(self, target_dims: np.ndarray, is_min: np.ndarray):
        """Adds dedicated directional search niches for specific scaled dimensions."""
        if self.pure_optimization:
            raise ValueError("Pure optimisation does not support extrema niches.")

        target_dims = np.asarray(target_dims, dtype=npint)
        is_min = np.asarray(is_min, dtype=np.bool_)

        if len(target_dims) != len(is_min):
            raise ValueError("Length of 'target_dims' and 'is_min' must match.")

        # Filter against global bounds to clamp valid dimensions
        valid_mask = (target_dims >= 0) & (target_dims < self.scaled_ndim)
        target_dims = target_dims[valid_mask]
        is_min = is_min[valid_mask]

        if len(target_dims) == 0:
            return

        # Deduplicate internally within the new request
        unique_pairs = np.unique(np.column_stack((target_dims, is_min)), axis=0)
        req_dims = unique_pairs[:, 0]
        req_is_min = unique_pairs[:, 1].astype(np.bool_)

        # Filter against already existing extrema niches
        existing_pairs = set()
        for i in range(self.num_niches):
            if self.niche_types[i] == 1:
                existing_pairs.add((self.niche_target_dims[i], self.niche_is_min[i]))

        novel_dims = []
        novel_is_min = []
        for d, m in zip(req_dims, req_is_min):
            if (d, m) not in existing_pairs:
                novel_dims.append(d)
                novel_is_min.append(m)

        if not novel_dims:
            return  # All requested niches already exist

        num_new = len(novel_dims)
        self.niche_types.extend([1] * num_new)
        self.niche_target_dims.extend(novel_dims)
        self.niche_is_min.extend(novel_is_min)
        self.num_niches += num_new

        if self._is_populated:
            self.population.add_extrema_niches(
                np.array(novel_dims, dtype=npint),
                np.array(novel_is_min, dtype=np.bool_)
            )

    def add_niches(
        self,
        diversity: int = 0,
        extrema_dims: np.ndarray[int] | str | None = None,
        extrema_is_min: np.ndarray[bool] | None = None,
    ):
        """
        Dispatches niche allocation based on kwargs.
        - diversity (int): Number of standard centroid-repulsion niches to add.
        - extrema_dims (array-like | str): Target scaled dimensions, or "all", "all-min", "all-max".
        - extrema_is_min (array-like): Boolean mask indicating if the extremum is a minimum.
        """
        if extrema_dims is not None:
            if isinstance(extrema_dims, str):
                d = self.scaled_ndim
                mode = extrema_dims.lower()
                if mode == "all":
                    extrema_dims = np.repeat(np.arange(d), 2).astype(npint)
                    extrema_is_min = np.tile([True, False], d).astype(np.bool_)
                    print(f"[MHMGA] Attempting to generate all {2 * d} extrema niches.")
                elif mode == "all-min":
                    extrema_dims = np.arange(d, dtype=npint)
                    extrema_is_min = np.ones(d, dtype=np.bool_)
                    print(f"[MHMGA] Attempting to generate all {d} minimizing extrema niches.")
                elif mode == "all-max":
                    extrema_dims = np.arange(d, dtype=npint)
                    extrema_is_min = np.zeros(d, dtype=np.bool_)
                    print(f"[MHMGA] Attempting to generate all {d} maximizing extrema niches.")
                else:
                    raise ValueError(f"Unknown string value for 'extrema_dims': {extrema_dims}")
            elif extrema_is_min is None:
                raise ValueError("Either a special string value for 'extrema_dims' or both"
                                 "'extrema_dims' and 'extrema_is_min' must be provided.")

            self.add_extrema_niches(extrema_dims, extrema_is_min)

        if diversity > 0:
            typing.sanitize_type(diversity, "integer", "diversity")
            self.add_diversity_niches(diversity)

    def update_hyperparameters(  # noqa: C901
        self,
        max_iter: int = _SENTINEL,
        pop_size: int = _SENTINEL,
        champ_count: int | float = _SENTINEL,
        elite_count: int | float = _SENTINEL,
        tourn_count: int | float = _SENTINEL,
        tourn_size: int = _SENTINEL,
        mutation_prob: float | tuple[float, float] = _SENTINEL,
        mutation_sigma: float | tuple[float, float] = _SENTINEL,
        mutation_alpha: float = _SENTINEL,
        crossover_prob: float | tuple[float, float] = _SENTINEL,
        violation_factor: float = _SENTINEL,
        noptimal_rel: float = _SENTINEL,
        noptimal_abs: float = _SENTINEL,
        niche_elitism: bool = _SENTINEL,
        mutation_scaler: np.ndarray | str = _SENTINEL,
        space_scaler: np.ndarray | str = _SENTINEL,
        objective_scaler: float = _SENTINEL,
        repulsion_weight: float = _SENTINEL,
        fitness_method: str = _SENTINEL,
        extrema_epsilon_rel: float = _SENTINEL,
        extrema_epsilon_abs: float = _SENTINEL,
        mark_as_set: bool = True
    ):
        updates = {k: v for k, v in locals().items() if v is not _SENTINEL and k != 'self'}

        if not updates:
            return  # Nothing to update

        # Update simple scalar parameters
        simple_attr_specs = (
            ("max_iter", "integer", None, 1),
            ("pop_size", "integer", None, 2),
            ("violation_factor", "float", None, 1),
            ("noptimal_rel", "float", None, 0),
            ("noptimal_abs", "float", None, 0),
            ("mutation_alpha", "float", None, None),
            ("repulsion_weight", "float", None, 0),
            ("extrema_epsilon_rel", "float", None, 0),
            ("extrema_epsilon_abs", "float", None, 0),
        )

        for (attr, expected_type, le, ge) in simple_attr_specs:
            if attr in updates:
                var = updates[attr]
                typing.sanitize_type(var, expected_type, attr)
                typing.sanitize_range(var, attr, le=le, ge=ge)
                clean_type = npint if expected_type == "integer" else npfloat
                setattr(self, attr, clean_type(var))

        # Tournament Size to be updated after pop_size
        if "tourn_size" in updates:
            ts = updates["tourn_size"]
            typing.sanitize_type(ts, "integer", "tourn_size")
            typing.sanitize_range(ts, "tourn_size", ge=2, lt=self.pop_size)
            self.tourn_size = npint(ts)

        # Update dithered probabilities
        for attr, ge, le in zip(
            ("mutation_prob", "mutation_sigma", "crossover_prob"),
            (0, None, 0), (1, None, 1)
        ):
            if attr in updates:
                val = typing.format_and_sanitize_ditherer(updates[attr], attr, npfloat, ge, le)
                setattr(self, attr, val)

        # Update Categoricals
        if "niche_elitism" in updates:
            val = updates["niche_elitism"]
            if val not in (None, True, False):
                raise ValueError(f"'niche_elitism' expected one of (None, True, False). Received: {val}")
            self.niche_elitism = bool(val)

        # Cascading Selection Counts
        if any(k in updates for k in ("champ_count", "elite_count", "tourn_count", "pop_size")):
            # Fetch explicitly provided updates, fallback to stored raw intents, fallback to standard defaults
            c_raw = updates.get("champ_count", getattr(self, "_raw_champ_count", 1))
            e_raw = updates.get("elite_count", getattr(self, "_raw_elite_count", 0.2))
            t_raw = updates.get("tourn_count", getattr(self, "_raw_tourn_count", -1))

            # Store raw intents so future pop_size updates recalculate correctly
            self._raw_champ_count, self._raw_elite_count, self._raw_tourn_count = c_raw, e_raw, t_raw

            for variable, name in zip((c_raw, e_raw, t_raw),
                                      ("champ_count", "elite_count", "tourn_count")):
                typing.sanitize_type(variable, ("float", "integer"), name)
                if typing.is_float(variable):
                    typing.sanitize_range(variable, name, ge=0, le=1)
                else:
                    typing.sanitize_range(variable, name, ge=-1, le=self.pop_size)

            champ_is_neg1 = c_raw == -1
            elite_is_neg1 = e_raw == -1
            tourn_is_neg1 = t_raw == -1

            if (champ_is_neg1 + elite_is_neg1 + tourn_is_neg1) > 1:
                raise ValueError("only one of 'champ_count', 'elite_count', and 'tourn_count' may be -1")

            c_int = npint(c_raw) if typing.is_integer(c_raw) else npint(c_raw * self.pop_size)
            e_int = npint(e_raw) if typing.is_integer(e_raw) else npint(e_raw * self.pop_size)
            t_int = npint(t_raw) if typing.is_integer(t_raw) else npint(t_raw * self.pop_size)

            if champ_is_neg1:
                c_int = self.pop_size - e_int - t_int
            if elite_is_neg1:
                e_int = self.pop_size - c_int - t_int
            if tourn_is_neg1:
                t_int = self.pop_size - c_int - e_int

            if c_int + e_int + t_int > self.pop_size:
                raise ValueError(f"champ_count + elite_count + tourn_count should be <= pop_size ({self.pop_size})")

            self.champ_count, self.elite_count, self.tourn_count = npint(c_int), npint(e_int), npint(t_int)

        # Scalers
        if "mutation_scaler" in updates:
            ms = updates["mutation_scaler"]
            if str(ms).lower() == "bounds":
                self.mutation_scaler = (self.problem.upper_bounds - self.problem.lower_bounds)
            elif ms is None:
                self.mutation_scaler = np.ones(self.problem.ndim, dtype=npfloat)
            else:
                typing.sanitize_array_type(ms, 'float', 'mutation_scaler', self.problem.ndim)
                self.mutation_scaler = np.array(ms, dtype=npfloat)

        if "space_scaler" in updates:
            ss = updates["space_scaler"]
            if self.problem.return_scaled:
                warnings.warn(
                    "[MHMGA] problem.return_scaled is set to True. 'space_scaler' will be ignored", UserWarning
                )
                new_ss = np.ones((1, self.problem.ndim), dtype=npfloat)
                new_scaled_ndim = self.problem.scaled_ndim
            elif ss is None:
                new_ss = np.ones((1, self.problem.ndim), dtype=npfloat)
                new_scaled_ndim = self.problem.ndim
            elif isinstance(ss, str):
                if ss.lower() == "bounds":
                    _scaler = 1.0 / (self.problem.upper_bounds - self.problem.lower_bounds)
                    new_ss = _scaler.reshape(1, self.problem.ndim).astype(npfloat)
                    new_scaled_ndim = self.problem.ndim
                else:
                    raise ValueError(f"Received unknown space_scaler. Expected an array or 'bounds'. Got: {ss}")
            else:
                new_ss = np.array(ss, dtype=npfloat)
                if new_ss.ndim == 1:
                    if new_ss.shape[0] != self.problem.ndim:
                        raise ValueError(f"1D space_scaler must have length {self.problem.ndim}")
                    new_ss = new_ss.reshape(1, self.problem.ndim)
                    new_scaled_ndim = self.problem.ndim
                elif new_ss.ndim == 2:
                    if new_ss.shape[0] != self.problem.ndim:
                        raise ValueError(f"2D space_scaler must have shape ({self.problem.ndim}, k)")
                    new_scaled_ndim = new_ss.shape[1]
                else:
                    raise ValueError("Array type space_scaler must be 1D or 2D")

            # Fail-fast guardrail: Prevents orphaned extrema niches
            if hasattr(self, "num_niches") and self.num_niches > 1:
                if new_scaled_ndim != self.scaled_ndim:
                    raise ValueError(
                        f"Cannot alter space_scaler dimensionality (from {self.scaled_ndim} to {new_scaled_ndim}) "
                        "after niches have been allocated. Please define space_scaler at initialization "
                        "or call update_hyperparameters before add_niches."
                    )

            self.space_scaler = new_ss

        if "objective_scaler" in updates:
            os = updates["objective_scaler"]
            if not self.include_obj_in_fitness:
                warnings.warn("[MHMGA] include_obj_in_fitness is False. 'objective_scaler' ignored", UserWarning)

            if os is None:
                self.objective_scaler = npfloat(1.0)
            else:
                typing.sanitize_type(os, 'float', 'objective_scaler')
                self.objective_scaler = npfloat(os)

        if "fitness_method" in updates:
            fm = updates["fitness_method"].lower()
            if fm not in ("angular", "l1", "l2"):
                raise ValueError(f"Unknown fitness method. Expected one of ('angular', 'l1', 'l2'). Got: '{fm}'")
            self.fitness_method = fm

        self.fitness_method_int = self.fitness_method_dict[self.fitness_method]
        self.hyperparameters_set = mark_as_set

    def reset_hyperparameters(self):
        """Restores all hyperparameters to their defaults."""
        self.update_hyperparameters(**self._DEFAULT_HYPERPARAMS)

    def _populate(
        self,
        x0: np.ndarray = None,
        points: np.ndarray = None,
        objectives: np.ndarray = None,
        constraints: np.ndarray = None,
        force: bool = False,
        evaluate: bool = True,
    ) -> None:
        """
        Generates and initializes the starting population.
        """
        if self._is_populated and not force:
            return
        if self.pop_size < 1:
            raise ValueError("'pop_size' must be positive definite.")
        if not self.hyperparameters_set:
            raise RuntimeError("Set hyperparameters before populating (`.update_hyperparameters`)")

        self._init_population_container()
        self._seed_population_points(x0, points)
        self._init_population_objectives(objectives, constraints, evaluate=evaluate)

        self._is_populated = True
        self._update_parent_size()

    def _init_population_container(self) -> None:
        """Instantiates the Population jitclass and binds problem & hyperparameters."""
        self.population = Population(
            num_niches=self.num_niches,
            pop_size=self.pop_size,
            ndim=self.problem.ndim,
            rng=self.rng,
            stable_sort=self.stable_sort,
            include_obj_in_fitness=self.include_obj_in_fitness,
            pure_optimization=self.pure_optimization,
            niche_types=np.array(self.niche_types, dtype=npint),
            niche_target_dims=np.array(self.niche_target_dims, dtype=npint),
            niche_is_min=np.array(self.niche_is_min, dtype=np.bool_),
        )
        load_problem_to_population(self.population, self.problem)
        self._update_population_hyperparameters()

    def _seed_population_points(self, x0: np.ndarray = None, points: np.ndarray = None) -> None:
        """Loads initial optimum x0 and populates decision variable points."""
        x0_given = x0 is not None and x0.size > 0 and x0.ndim > 0
        points_given = points is not None and points.size > 0 and points.ndim > 0
        _optimum = -np.inf if self.problem.maximize else np.inf

        if x0_given:
            _optimum_point = x0
        elif points_given:
            _optimum_point = points[0] if points.ndim == 2 else points[0, 0]
        else:
            _optimum_point = np.empty([], npfloat)

        self.population._load_optimum(_optimum_point, _optimum, 0.0, 0.0)

        if points_given:
            pop_seed = points
        elif x0_given:
            pop_seed = x0
        else:
            pop_seed = np.empty([], npfloat)

        self.population.initialize_population(pop_seed)

        if not self.problem.constraints:
            self.population.violations[:] = 0.0

    def _init_population_objectives(
        self,
        objectives: np.ndarray = None,
        constraints: np.ndarray = None,
        evaluate: bool = True,
    ) -> None:
        """Evaluates population or loads pre-evaluated objectives/constraints, then updates fitness & optima."""
        has_preloaded = (
            objectives is not None
            and (not self.problem.constraints or constraints is not None)
        )

        if not evaluate and has_preloaded and not self.problem.return_scaled:
            self.population.raw_objectives[:] = objectives.astype(npfloat)
            if self.problem.constraints:
                self.population.violations[:] = constraints.astype(npfloat)
            self.population.penalized_objectives[:] = (
                self.population.raw_objectives + self.population.violations * self.violation_factor
            )
        else:
            if not evaluate and self.problem.return_scaled:
                warnings.warn(
                    "[MHMGA] Cannot skip evaluation when problem.return_scaled is True. Re-evaluating.",
                    UserWarning,
                )
            elif not evaluate and not has_preloaded:
                warnings.warn(
                    "[MHMGA] evaluate=False requested but objectives/constraints missing. Re-evaluating.",
                    UserWarning,
                )
            self._evaluate_points()

        self.population.evaluate_fitness()
        self.population.track_optima()

    def set_verbosity(self, disp_rate=None, verbose=None):
        if disp_rate is not None:
            typing.sanitize_type(disp_rate, "integer", "disp_rate")
            typing.sanitize_range(disp_rate, "disp_rate", ge=-1)
            self.disp_rate = disp_rate
        if verbose is not None:
            self.verbosity = verbose

    def step(  # noqa: C901
        self,
        convergence_criteria: None | term.Convergence | list[term.Convergence] = None,
        evaluate_initial: bool | None = None
    ):
        """
        Executes the main optimization loop.
        """
        if not self.hyperparameters_set:
            raise RuntimeError("Set hyperparameters before launching a step (`.update_hyperparameters`)")

        termination_handler = self.configure_termination(convergence_criteria)
        # Instantiation
        if not self._is_populated:
            if not hasattr(self, "num_niches") or self.num_niches < 1:
                raise RuntimeError(f"'num_niches' must be greater than zero. Got: {self.num_niches}"
                                   ". Call `.add_niches()` first.")
            if evaluate_initial is None:
                evaluate_initial = self.starting_objectives is None

            self._populate(
                x0=self.x0,
                points=self.starting_points,
                objectives=self.starting_objectives,
                constraints=self.starting_constraints,
                evaluate=evaluate_initial,
            )
            if self.callback is not None and evaluate_initial:
                self.callback(self.population)
        else:
            self._update_population_hyperparameters()

        # Main algorithm loop
        self._stop_requested = False
        try:
            while not termination_handler(self):
                self._display_progress()

                self._run_iteration()

                if self.logger:
                    self.logger.log_iteration(self.current_iter, self.population)

                # provides hooks for termination control
                # TODO: more hooks
                self.current_best_obj = self.population.optima_raw_objectives[0]
                self.mean_fitness = self.population.mean_fitness

                self.current_iter += 1

                if self.callback is not None:
                    self.callback(self.population)

        except (KeyboardInterrupt, SystemError) as exc:
            if isinstance(exc, SystemError) and not getattr(self, "_stop_requested", False):
                raise exc
            print("Received Keyboard Interrupt. Terminating gracefully.")
            self._stop_requested = True

        self._display_progress()

    def inspect_recombination(
        self,
        starting_points: np.ndarray,
        point_objectives: np.ndarray = None,
        point_constraints: np.ndarray = None,
        evaluate_offspring: bool = True,
    ) -> dict:
        """
        Executes a single, isolated round of evaluation and recombination.
        """
        if not self.hyperparameters_set:
            raise RuntimeError("Set hyperparameters before tuning (`.update_hyperparameters`)")
        if not self.num_niches == 1:
            raise NotImplementedError("inspect_recombination is only implemented for single-niche populations")

        starting_points = self._sanitize_seed_points(starting_points, "starting_points", check_all_dims=True)

        obj_given = point_objectives is not None
        cons_given = point_constraints is not None

        if self.problem.constraints:
            evaluate = not (obj_given and cons_given)
            if obj_given != cons_given:
                warnings.warn(
                    f"starting point objectives {'' if obj_given else 'not'} supplied"
                    f" but constraint violations {'' if cons_given else 'not'} supplied."
                    "Points will be re-evaluated",
                    UserWarning
                )
        else:
            cons_given = False  # ignore constraint inputs if problem has no constraints
            evaluate = point_objectives is None

        self._populate(
            points=starting_points,
            objectives=point_objectives,
            constraints=point_constraints,
            force=True,
            evaluate=evaluate,
        )

        starting_points = self.population.points[0, :, :].copy()  # standardise output to 2D
        point_objectives = self.population.raw_objectives[0, :].copy()
        point_constraints = self.population.violations[0, :].copy()

        self.population.dither_probabilities()
        self.population.select_parents()
        self.population.generate_offspring()

        if evaluate_offspring:
            self._evaluate_points()

        return {
            "parents_points": starting_points,
            "parents_objectives": point_objectives,
            "parents_violations": point_constraints,

            "offspring_points": self.population.points[0, :, :].copy(),
            "offspring_objectives": self.population.raw_objectives[0, :].copy() if evaluate_offspring else None,
            "offspring_violations": self.population.violations[0, :].copy() if evaluate_offspring else None,
        }

    def get_results(self) -> dict:
        """
        Returns the final results of the optimization.
        """
        if self.logger:
            self.logger.finalize(self.population)

        if self.population is None:
            raise RuntimeError("Algorithm has not been run yet.")

        return {
            "optima": self.population.optima_points,
            "fitness": self.population.optima_fitnesses,
            "objective": self.population.optima_raw_objectives,
            "penalties": self.population.optima_penalized_objectives - self.population.optima_raw_objectives,
            "noptimality": self.population.optima_noptimal_mask,
        }

    def _update_parent_size(self):
        if not self._is_populated:
            return
        if not self.hyperparameters_set:
            return
        self.population.resize(
            -1,  # ignore niches
            self.pop_size,  # pop_size
            self.champ_count + self.elite_count + self.tourn_count,  # parent_size
        )

    def _sanitize_seed_points(self, points, name="points", check_all_dims=False):
        if points is None or points.size == 0:
            return np.empty([], npfloat)

        typing.sanitize_type(points, np.ndarray, name)

        if check_all_dims:
            if points.ndim == 3 and points.shape[0] == 1:
                points = points[0]
            if points.ndim == 2 and points.shape[0] == 1 and self.pop_size == 1:
                points = points[0]

            expected_shape = {
                1: (self.problem.ndim,),
                2: (self.pop_size, self.problem.ndim),
                3: (self.num_niches, self.pop_size, self.problem.ndim)
            }

            shape_str = {
                1: f"(ndim={self.problem.ndim},)",
                2: f"(pop_size={self.pop_size}, ndim={self.problem.ndim})",
                3: f"(num_niches={self.num_niches}, pop_size={self.pop_size}, ndim={self.problem.ndim})"
            }

            if points.shape != expected_shape[points.ndim]:
                raise ValueError(
                    f"Bad shape for '{name}'. Expected {shape_str[points.ndim]}, got {points.shape}."
                )
        else:
            if points.shape[-1] != self.problem.ndim:
                raise ValueError(
                    f"Last dimension of '{name}' should be ndim={self.problem.ndim}. Got shape {points.shape}."
                )

        return points.astype(npfloat)

    def _run_iteration(self):
        """
        Dispatcher method.
        Runs one iteration using the fastest available path.
        """
        if self._can_use_fast_path:
            _njit_iteration_loop(
                self.population,
                self.evaluator,
                self.problem.objective,
                self.problem.fargs,
            )
        else:
            _python_iteration_loop(
                self.population,
                self.evaluator,
                self.problem.objective,
                self.problem.fargs,
                self.problem.fkwargs
            )

    def _evaluate_points(self):
        if self._can_use_fast_path:
            self.evaluator(
                self.population, self.problem.objective, self.problem.fargs
            )
        else:
            self.evaluator(
                self.population, self.problem.objective, self.problem.fargs, self.problem.fkwargs
            )

    def _update_population_hyperparameters(self):
        # numba does not always like keyword arguments
        self.population.update_hyperparameters(
            self.champ_count,
            self.elite_count,
            self.tourn_count,
            self.tourn_size,
            self.mutation_prob,
            self.mutation_sigma,
            self.mutation_alpha,
            self.crossover_prob,
            self.niche_elitism,
            self.noptimal_rel,
            self.noptimal_abs,
            self.violation_factor,
            self.mutation_scaler,
            self.space_scaler,
            self.objective_scaler,
            self.repulsion_weight,
            self.fitness_method_int,
            self.extrema_epsilon_rel,
            self.extrema_epsilon_abs,
        )
        self._update_parent_size()

    def _display_progress(self):
        """
        Prints the current progress of the algorithm to the console.
        """
        if self.disp_rate == 0:
            return
        if self.current_iter % self.disp_rate != 0:
            return
        if self.verbosity == 0:
            return

        best_pobj = self.population.optima_penalized_objectives[0]
        elapsed = dt.now() - self.start_time

        iter_str = f"Iter: {self.current_iter}. "
        time_str = f"Time: {elapsed}. "

        if self.verbosity >= 1:
            print(iter_str + time_str, end="")

        penobj_str = f"Best (pen) Objective: {best_pobj:.2f}"
        infeas_str = " [infeasible]. " if self.population.optima_violations[0] > 0 else ". "

        if self.verbosity >= 2:
            print(penobj_str + infeas_str, end="")

        dith_str = (f"mutation_prob: {self.population.current_mutation_prob:.4f}, "
                    f"mutation_sigma: {self.population.current_mutation_sigma:.4f}, "
                    f"crossover_prob: {self.population.current_crossover_prob:.4f}. ")

        if self.verbosity >= 3:
            print(dith_str, end="")

        print("\n", end="")

    def configure_termination(self, convergence_criteria):
        typing.sanitize_type(
            convergence_criteria, (term.Convergence, "arraylike", "none"), "convergence_criteria"
        )
        if typing.is_array_like(convergence_criteria):
            typing.sanitize_array_type(convergence_criteria, term.Convergence, "convergence_criteria")

        if convergence_criteria is None:
            termination_handler = term.Maxiter(self.max_iter)
        elif hasattr(convergence_criteria, "__iter__"):
            termination_handler = term.MultiConvergence([term.Maxiter(self.max_iter), *convergence_criteria])
        else:
            termination_handler = term.MultiConvergence([term.Maxiter(self.max_iter), convergence_criteria])

        return termination_handler


def _python_iteration_loop(
    population: Population,
    eval_func: Callable,
    objective_func: Callable,
    fargs: tuple,
    fkwargs: dict,
) -> None:
    """
    The fallback "slow path" for non-jitted objectives or
    objectives with fargs/fkwargs.
    """
    population.dither_probabilities()
    population.select_parents()
    population.generate_offspring()
    eval_func(population, objective_func, fargs, fkwargs)
    population.evaluate_fitness()  # TODO: hooks for termination
    population.track_optima()
    population.evaluate_diversity()  # TODO: hooks for termination


@njit
def _njit_iteration_loop(
    population: Population,
    eval_func: CPUDispatcher,
    objective_func: CPUDispatcher,
    fargs: tuple,
) -> None:
    population.dither_probabilities()
    population.select_parents()
    population.generate_offspring()
    eval_func(population, objective_func, fargs)
    population.evaluate_fitness()  # TODO: hooks for termination
    population.track_optima()
    population.evaluate_diversity()  # TODO: hooks for termination
