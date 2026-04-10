import numpy as np
from numpy.typing import NDArray
from datetime import datetime as dt
import warnings
from numba.core.registry import CPUDispatcher
from typing import Callable

from mga.commons.types import npint, npfloat
from mga.commons.numba_overload import njit, prange
import mga.utils.termination as term
from mga.utils import typing
from mga.problem_definition import OptimizationProblem
from mga.population import Population, load_problem_to_population
from mga.utils.logger import Logger


_SENTINEL = object()


class MGAProblem:
    """
    Orchestrates the modelling to generate alternatives algorithm.
    Manages the optimization loop, problem state, and logging.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        x0: np.ndarray[float] = None,
        log_dir: str | None = None,
        log_freq: int = -1,
        random_seed: int | None = None,
        parallelize: bool = True,
        callback: Callable = None,
        include_obj_in_fitness: bool = False,
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

        # Instantiation
        self.problem = problem
        self.rng = np.random.default_rng(random_seed)
        self.stable_sort = random_seed is not None
        self.logger = Logger(log_dir, log_freq, create_dir=True, ndim=self.problem.ndim) if log_dir else None

        self.x0 = self._sanitize_seed_points(x0, "x0")

        self.population = None
        self.current_iter = 0
        self.start_time = dt.now()

        self.niche_elitism_dict = {
            None: npint(0),
            "selfish": npint(1),
            "unselfish": npint(2),
        }

        # State and hyperparameter storage
        self._is_populated = False
        self.pop_size = npint(0)
        self.elite_count = npint(0)
        self.tourn_count = npint(0)
        self.tourn_size = npint(0)
        self.mutation_prob = npfloat(0.0)
        self.mutation_sigma = npfloat(0.0)
        self.crossover_prob = npfloat(0.0)
        self.niche_elitism = None
        self.noptimal_rel = npfloat(0.0)
        self.noptimal_abs = npfloat(0.0)
        self.mutation_scaler = None
        self.space_scaler = None
        self.current_best_obj = npfloat(0)
        self.mean_fitness = npfloat(0)
        self.hyperparameters_set = False
        self.callback = callback
        self.include_obj_in_fitness = include_obj_in_fitness

        # Evaluation Paths
        self._can_use_fast_path = self.problem.objective_jitted and not self.problem.fkwargs
        if self._can_use_fast_path:
            self.evaluator = construct_njit_eval_func(
                self.problem.vectorized,
                self.problem.constraints,
                self.problem.return_scaled,
                parallelize,
            )
            print("--- MHMGA: Njitted objective detected. Using fast-path iteration loop. ---")
        else:
            self.evaluator = construct_python_eval_func(
                self.problem.vectorized,
                self.problem.constraints,
                self.problem.return_scaled,
            )
            print("--- MHMGA: Using standard Python iteration loop (objective is not njitted or uses fkwargs). ---")

    def add_niches(self, num_niches: int):
        """
        Adds niches to the problem
        """
        typing.sanitize_type(num_niches, "integer", "num_niches")
        typing.sanitize_range(num_niches, "num_niches", ge=1)

        if not self._is_populated:
            self.num_niches = num_niches
        else:
            self.population.add_niches(num_niches)
            self.num_niches += num_niches

    def update_hyperparameters(  # noqa: C901
            self,
            max_iter: int = 0,
            pop_size: int = 100,
            elite_count: int | float = 0.2,
            tourn_count: int | float = 0.8,
            tourn_size: int = 2,
            mutation_prob: float | tuple[float, float] = 0.3,
            mutation_sigma: float | tuple[float, float] = 0.05,
            crossover_prob: float | tuple[float, float] = 0.4,
            violation_factor: float = 1.0,
            noptimal_rel: float = 0.0,
            noptimal_abs: float = 0.0,
            niche_elitism: str = "selfish",
            mutation_scaler: NDArray = "bounds",
            space_scaler: NDArray = _SENTINEL,  # default: 'bounds'
            objective_scaler: float = _SENTINEL,  # default: 1.0
            ):
        typing.sanitize_type(max_iter, "integer", "max_iter")
        typing.sanitize_range(max_iter, "max_iter", ge=1)
        self.max_iter = npint(max_iter)

        typing.sanitize_type(pop_size, "integer", "pop_size")
        typing.sanitize_range(pop_size, "pop_size", ge=2)
        self.pop_size = npint(pop_size)

        typing.sanitize_type(elite_count, ("float", "integer"), "elite_count")
        if typing.is_float(elite_count):
            typing.sanitize_range(elite_count, "elite_count", ge=0, le=1)
        else:
            typing.sanitize_range(elite_count, "elite_count", ge=-1, le=pop_size)
        typing.sanitize_type(tourn_count, ("float", "integer"), "tourn_count")
        if typing.is_float(tourn_count):
            typing.sanitize_range(tourn_count, "tourn_count", ge=0, le=1)
        else:
            typing.sanitize_range(tourn_count, "tourn_count", ge=-1, le=pop_size)

        typing.sanitize_type(tourn_size, "integer", "tourn_size")
        typing.sanitize_range(tourn_size, "tourn_size", gt=1, lt=pop_size)
        self.tourn_size = npint(tourn_size)

        self.mutation_prob = typing.format_and_sanitize_ditherer(mutation_prob, "mutation_prob", npfloat, 0, 1)
        self.mutation_sigma = typing.format_and_sanitize_ditherer(mutation_sigma, "mutation_sigma", npfloat)
        self.crossover_prob = typing.format_and_sanitize_ditherer(crossover_prob, "crossover_prob", npfloat, 0, 1)

        typing.sanitize_type(violation_factor, "float", "violation_factor")
        typing.sanitize_range(violation_factor, "violation_factor", ge=1)
        self.violation_factor = npfloat(violation_factor)

        typing.sanitize_type(noptimal_rel, "float", "noptimal_rel")
        typing.sanitize_range(noptimal_rel, "noptimal_rel", ge=0)
        self.noptimal_rel = npfloat(noptimal_rel)

        typing.sanitize_type(noptimal_abs, "float", "noptimal_abs")
        typing.sanitize_range(noptimal_abs, "noptimal_abs", ge=0)
        self.noptimal_abs = npfloat(noptimal_abs)

        if niche_elitism not in (None, "selfish", "unselfish"):
            raise ValueError(
                f"'niche_elitism' expected one of (None, 'selfish', 'unselfish'). Received: {niche_elitism}"
            )
        self.niche_elitism = niche_elitism
        self.niche_elitism_int = self.niche_elitism_dict[self.niche_elitism]

        if elite_count == -1 and tourn_count == -1:
            raise ValueError("only 1 of 'elite_count' and 'tourn_count' may be -1")
        elite_count = npint(elite_count) if typing.is_integer(elite_count) else npint(elite_count * pop_size)
        tourn_count = npint(tourn_count) if typing.is_integer(tourn_count) else npint(tourn_count * pop_size)
        elite_count = pop_size - tourn_count if elite_count == -1 else elite_count
        tourn_count = pop_size - elite_count if tourn_count == -1 else tourn_count
        if elite_count + tourn_count > pop_size:
            raise ValueError("'elite_count' + 'tourn_count' should be <= 'pop_size'")
        self.elite_count = npint(elite_count)
        self.tourn_count = npint(tourn_count)

        if str(mutation_scaler).lower() == "bounds":
            self.mutation_scaler = (self.problem.upper_bounds - self.problem.lower_bounds)
        elif mutation_scaler is None:
            self.mutation_scaler = np.ones(self.problem.ndim, dtype=npfloat)
        else:
            typing.sanitize_array_type(mutation_scaler, 'float', 'mutation_scaler', self.problem.ndim)
            self.mutation_scaler = np.array(mutation_scaler, dtype=npfloat)

        space_scaler_given = space_scaler is not _SENTINEL
        if space_scaler_given and self.problem.return_scaled:
            warnings.warn(
                "[MHMGA] problem.return_scaled is set to True. 'space_scaler' will be ignored", UserWarning
            )

        if (str(space_scaler).lower() == "bounds") or not space_scaler_given:
            self.space_scaler = (self.problem.upper_bounds - self.problem.lower_bounds)
        elif space_scaler is None:
            self.space_scaler = np.ones(self.problem.ndim, dtype=npfloat)
        else:
            typing.sanitize_array_type(space_scaler, 'float', 'space_scaler', self.problem.ndim)
            self.space_scaler = np.array(space_scaler, dtype=npfloat)

        objective_scaler_given = objective_scaler is not _SENTINEL
        if objective_scaler_given and not self.include_obj_in_fitness:
            warnings.warn(
                "[MHMGA] include_obj_in_fitness is set to False. 'objective_scaler' will be ignored", UserWarning
            )
        if (not self.include_obj_in_fitness) or (not objective_scaler_given):
            self.objective_scaler = npfloat(1.0)
        else:
            typing.sanitize_type(objective_scaler, 'float', 'objective_scaler')
            self.objective_scaler = npfloat(objective_scaler)

        self.hyperparameters_set = True

    def _populate(
        self,
        points: np.ndarray = None,
        force: bool = False,
        evaluate: bool = True,
    ) -> None:
        """
        generate starting population
        """
        if self._is_populated and not force:
            return
        if self.pop_size < 1:
            raise ValueError("'pop_size' must be positive definite.")

        if not self.hyperparameters_set:
            raise RuntimeError("Set hyperparameters before populating (`.update_hyperparameters`)")

        # Initialize population
        self.population = Population(
            num_niches=self.num_niches,
            pop_size=self.pop_size,
            ndim=self.problem.ndim,
            rng=self.rng,
            stable_sort=self.stable_sort,
            include_obj_in_fitness=self.include_obj_in_fitness,
        )
        load_problem_to_population(self.population, self.problem)

        self._update_population_hyperparameters()
        if points is None:
            self.population.initialize_population(self.noptimal_rel, self.noptimal_abs, self.violation_factor, self.x0)
        else:
            self.population.initialize_population(self.noptimal_rel, self.noptimal_abs, self.violation_factor, points)

        if not self.problem.constraints:
            self.population.violations[:] = 0.0

        if evaluate:
            self._evaluate_points()
            self.population.evaluate_fitness()
            self.population.track_optima()

        self._is_populated = True

        self._update_parent_size()

    def step(  # noqa: C901
        self,
        disp_rate: int = 0,
        convergence_criteria: None | term.Convergence | list[term.Convergence] = None,
    ):
        """
        Executes the main optimization loop.
        """
        if not self.hyperparameters_set:
            raise RuntimeError("Set hyperparameters before launching a step (`.update_hyperparameters`)")
        typing.sanitize_type(disp_rate, "integer", "disp_rate")
        typing.sanitize_range(disp_rate, "disp_rate", ge=-1)

        termination_handler = self.configure_termination(convergence_criteria)

        # Instantiation
        if not self._is_populated:
            if not hasattr(self, "num_niches") or self.num_niches < 1:
                raise RuntimeError(f"'num_niches' must be greater than zero. Got: {self.num_niches}"
                                   ". Call `.add_niches()` first.")
            self._populate()
        else:
            self._update_population_hyperparameters()

        # Main algorithm loop
        try:
            while not termination_handler(self):
                if disp_rate > 0 and self.current_iter % disp_rate == 0:
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

        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Terminating gracefully.")
            pass

        if disp_rate != 0:
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

        self._populate(starting_points, force=True, evaluate=evaluate)
        if not evaluate:
            self.population.raw_objectives[0, :] = point_objectives.astype(npfloat)
            self.population.violations[0, :] = point_constraints.astype(npfloat) if cons_given else npfloat(0.0)
            self.population.penalized_objectives[:] = (
                self.population.raw_objectives + self.population.violations * self.violation_factor
            )
            self.population.evaluate_fitness()
            self.population.track_optima()

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
            self.tourn_count + self.elite_count,  # parent_size
        )

    def _sanitize_seed_points(self, points, name="points", check_all_dims=False):
        if points is None or points.size == 0:
            return np.empty([], npfloat)

        typing.sanitize_type(points, np.ndarray, name)

        # Flatten if num_niches or pop_size is 1
        if points.shape[0] == 1:
            points = points[0]
        if points.shape[0] == 1:
            points = points[0]

        if check_all_dims:
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

        points = points.astype(npfloat)
        return points

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
        # numba does not like keyword arguments
        self.population.update_hyperparameters(
            self.elite_count,
            self.tourn_count,
            self.tourn_size,
            self.mutation_prob,
            self.mutation_sigma,
            self.crossover_prob,
            self.niche_elitism_int,
            self.noptimal_rel,
            self.noptimal_abs,
            self.violation_factor,
            self.mutation_scaler,
            self.space_scaler,
            self.objective_scaler
        )

    def _display_progress(self):
        """
        Prints the current progress of the algorithm to the console.
        """
        best_pobj = self.population.optima_penalized_objectives[0]
        elapsed = dt.now() - self.start_time

        if self.population.optima_violations[0] > 0:
            print(f"Iter: {self.current_iter}. Best Objective: {best_pobj:.2f} [infeasible]. Time: {elapsed}")
        else:
            print(f"Iter: {self.current_iter}. Best Objective: {best_pobj:.2f}. Time: {elapsed}")

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


def construct_python_eval_func(  # noqa: C901
        vectorized: bool,
        constraints: bool,
        return_scaled: bool,
) -> Callable:
    """
    Constructs the appropriate evaluation function based on problem properties.
    """
    if vectorized and constraints and return_scaled:
        def _evaluator(population: Population, objective_func: Callable, fargs: tuple, fkwargs: dict):
            """
            Evaluates a vectorized objective function that returns obj_vals, violation_vals, and scaled_points.
            """
            for i in range(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.violations[i, :],
                    population.scaled_points[i, :, :]
                 ) = objective_func(population.points[i], *fargs, **fkwargs)

            population.penalized_objectives[:] = (
                population.raw_objectives + population.violations * population.violation_factor
            )

    elif vectorized and not constraints and return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a vectorized objective function that returns obj_vals and scaled_points.
            """
            for i in range(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.scaled_points[i, :, :],
                ) = objective_func(population.points[i], *fargs, **fkwargs)

            population.penalized_objectives[:] = population.raw_objectives  # No penalty

    elif not vectorized and constraints and return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val, and scaled_points
            """
            for i in range(population.num_niches):
                for j in range(population.pop_size):
                    (
                        population.raw_objectives[i, j],
                        population.violations[i, j],
                        population.scaled_points[i, j, :]
                    ) = objective_func(population.points[i, j], *fargs, **fkwargs)

            population.penalized_objectives[:] = (
                population.raw_objectives + population.violations * population.violation_factor
            )

    elif not vectorized and not constraints and return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a scalar objective function that returns obj_val and scaled_points
            """
            for i in range(population.num_niches):
                for j in range(population.points[i].shape[0]):
                    (
                        population.raw_objectives[i, j],
                        population.scaled_points[i, j, :],
                    ) = objective_func(population.points[i, j], *fargs, **fkwargs)

            population.penalized_objectives[:] = population.raw_objectives  # No penalty

    elif vectorized and constraints and not return_scaled:
        def _evaluator(population: Population, objective_func: Callable, fargs: tuple, fkwargs: dict):
            for i in range(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.violations[i, :]
                ) = objective_func(population.points[i], *fargs, **fkwargs)

            population.penalized_objectives[:] = (
                population.raw_objectives + population.violations * population.violation_factor
            )

    elif vectorized and not constraints and not return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            for i in range(population.num_niches):
                population.raw_objectives[i, :] = objective_func(population.points[i], *fargs, **fkwargs)

            population.penalized_objectives[:] = population.raw_objectives  # No penalty

    elif not vectorized and constraints and not return_scaled:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val.
            """
            for i in range(population.num_niches):
                for j in range(population.pop_size):
                    (
                        population.raw_objectives[i, j],
                        population.violations[i, j]
                    ) = objective_func(population.points[i, j], *fargs, **fkwargs)

            population.penalized_objectives[:] = (
                population.raw_objectives + population.violations * population.violation_factor
            )

    else:  # not vectorized and not constraints and not return_scaled
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a scalar objective function that returns only obj_val.
            """
            for i in range(population.num_niches):
                for j in range(population.points[i].shape[0]):
                    population.raw_objectives[i, j] = objective_func(population.points[i, j], *fargs, **fkwargs)

            population.penalized_objectives[:] = population.raw_objectives  # No penalty

    return _evaluator


def construct_njit_eval_func(  # noqa: C901
        vectorized: bool,
        constraints: bool,
        return_scaled: bool,
        parallelize: bool,
) -> CPUDispatcher:
    """
    Constructs the appropriate evaluation function based on problem properties.
    """
    if vectorized and constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns obj_vals, violation_vals.
            """
            for i in prange(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.violations[i, :],
                    population.scaled_points[i, :, :],
                ) = objective_func(population.points[i], *fargs)

                for j in range(population.pop_size):
                    population.penalized_objectives[i, j] = (
                        population.raw_objectives[i, j]
                        + population.violations[i, j] * population.violation_factor
                    )

    elif vectorized and not constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            for i in prange(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.scaled_points[i, :, :]
                ) = objective_func(population.points[i], *fargs)

                for j in range(population.pop_size):
                    population.penalized_objectives[i, j] = population.raw_objectives[i, j]

    elif not vectorized and constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.pop_size):
                    (
                        population.raw_objectives[i, j],
                        population.violations[i, j],
                        population.scaled_points[i, j, :],
                    ) = objective_func(population.points[i, j], *fargs)

                    population.penalized_objectives[i, j] = (
                        population.raw_objectives[i, j]
                        + population.violations[i, j] * population.violation_factor
                    )

    elif not vectorized and not constraints and return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns only obj_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.pop_size):
                    (
                        population.raw_objectives[i, j],
                        population.scaled_points[i, j, :],
                    ) = objective_func(population.points[i, j], *fargs)

                    population.penalized_objectives[i, j] = population.raw_objectives[i, j]

    elif vectorized and constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns obj_vals, violation_vals.
            """
            for i in prange(population.num_niches):
                (
                    population.raw_objectives[i, :],
                    population.violations[i, :]
                ) = objective_func(population.points[i], *fargs)

                for j in range(population.pop_size):
                    population.penalized_objectives[i, j] = (
                        population.raw_objectives[i, j]
                        + population.violations[i, j] * population.violation_factor
                    )

    elif vectorized and not constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            for i in prange(population.num_niches):
                population.raw_objectives[i, :] = objective_func(population.points[i], *fargs)

                for j in range(population.pop_size):
                    population.penalized_objectives[i, j] = population.raw_objectives[i, j]

    elif not vectorized and constraints and not return_scaled:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.pop_size):
                    (
                        population.raw_objectives[i, j],
                        population.violations[i, j]
                    ) = objective_func(population.points[i, j], *fargs)

                    population.penalized_objectives[i, j] = (
                        population.raw_objectives[i, j]
                        + population.violations[i, j] * population.violation_factor
                    )

    else:  # not vectorized and not constraints and not return_scaled
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns only obj_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.points[i].shape[0]):
                    population.raw_objectives[i, j] = objective_func(population.points[i, j], *fargs)

                    population.penalized_objectives[i, j] = population.raw_objectives[i, j]

    return _evaluator
