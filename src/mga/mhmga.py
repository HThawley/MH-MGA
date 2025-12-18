import numpy as np
from datetime import datetime as dt
import warnings
from numba import njit, prange
from numba.core.registry import CPUDispatcher
from typing import Callable

from mga.commons.types import DEFAULTS

INT, FLOAT = DEFAULTS
import mga.utils.termination as term  # noqa: E402
from mga.utils import typing  # noqa: E402
from mga.problem_definition import OptimizationProblem  # noqa: E402
from mga.population import Population, load_problem_to_population  # noqa: E402
from mga.utils.logger import Logger  # noqa: E402


class MGAProblem:
    """
    Orchestrates the modelling to generate alternatives algorithm.
    Manages the optimization loop, problem state, and logging.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        x0: np.ndarray[float] = np.empty(0, FLOAT),
        log_dir: str | None = None,
        log_freq: int = -1,
        random_seed: int | None = None,
        parallelize: bool = True,
        callback: Callable = None,
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

        # Instantiation
        self.problem = problem
        self.rng = np.random.default_rng(random_seed)
        self.stable_sort = random_seed is not None
        self.logger = Logger(log_dir, log_freq, create_dir=True) if log_dir else None

        if x0 is None:
            x0 = np.empty(0, FLOAT)
        typing.sanitize_type(x0, np.ndarray, "x0")
        if x0.size > 0:
            if not x0.dtype != FLOAT:
                warnings.warn("x0 will be cast to float")
            if not x0.ndim == 1:
                raise NotImplementedError("currently only 1-dimensional 'x0' is accepted")
            if not len(x0) == self.problem.ndim:
                raise ValueError("'x0' should be of same length as 'problem.ndim'")
        self.x0 = x0

        self.population = None
        self.current_iter = INT(0)
        self.start_time = dt.now()

        self.niche_elitism_dict = {
            None: 0,
            "selfish": 1,
            "unselfish": 2,
        }

        # State and hyperparameter storage
        self._is_populated = False
        self.pop_size = INT(0)
        self.elite_count = INT(0)
        self.tourn_count = INT(0)
        self.tourn_size = INT(0)
        self.mutation_prob = FLOAT(0.0)
        self.mutation_sigma = FLOAT(0.0)
        self.crossover_prob = FLOAT(0.0)
        self.niche_elitism = None
        self.noptimal_slack = FLOAT(1.0)
        self.current_best_obj = FLOAT(0)
        self.mean_fitness = FLOAT(0)
        self.hyperparameters_set = False
        self.callback = callback

        # Evaluation Paths
        self._can_use_fast_path = self.problem.objective_jitted and not self.problem.fkwargs
        if self._can_use_fast_path:
            self.evaluator = construct_njit_eval_func(
                self.problem.vectorized,
                self.problem.constraints,
                parallelize,
            )
            print("--- MHMGA: Njitted objective detected. Using fast-path iteration loop. ---")
        else:
            self.evaluator = construct_python_eval_func(
                self.problem.vectorized,
                self.problem.constraints
            )
            print("--- MHMGA: Using standard Python iteration loop (objective is not njitted or uses fkwargs). ---")

    def add_niches(self, num_niches: int):
        """
        Adds niches to the problem
        """
        typing.sanitize_type(num_niches, "integer", "num_niches")
        typing.sanitize_range(num_niches, "num_niches", gt=1)

        if not self._is_populated:
            self.num_niches = num_niches
        else:
            self.population.add_niches(num_niches)
            self.num_niches += num_niches

    def update_hyperparameters(
            self,
            max_iter: int,
            pop_size: int = 100,
            elite_count: int | float = 0.2,
            tourn_count: int | float = 0.8,
            tourn_size: int = 2,
            mutation_prob: float | tuple[float, float] = 0.3,
            mutation_sigma: float | tuple[float, float] = 0.05,
            crossover_prob: float | tuple[float, float] = 0.4,
            violation_factor: float = 1.0,
            noptimal_slack: float = np.inf,
            niche_elitism: str = "selfish",
            ):
        typing.sanitize_type(max_iter, "integer", "max_iter")
        typing.sanitize_range(max_iter, "max_iter", ge=1)
        self.max_iter = INT(max_iter)

        typing.sanitize_type(pop_size, "integer", "pop_size")
        typing.sanitize_range(pop_size, "pop_size", ge=2)
        self.pop_size = INT(pop_size)

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
        self.tourn_size = INT(tourn_size)

        self.mutation_prob = typing.format_and_sanitize_ditherer(mutation_prob, "mutation_prob", FLOAT, 0, 1)
        self.mutation_sigma = typing.format_and_sanitize_ditherer(mutation_sigma, "mutation_sigma", FLOAT)
        self.crossover_prob = typing.format_and_sanitize_ditherer(crossover_prob, "crossover_prob", FLOAT, 0, 1)

        typing.sanitize_type(violation_factor, "float", "violation_factor")
        typing.sanitize_range(violation_factor, "violation_factor", ge=0)
        self.violation_factor = FLOAT(violation_factor)

        typing.sanitize_type(noptimal_slack, "float", "noptimal_slack")
        typing.sanitize_range(noptimal_slack, "noptimal_slack", ge=0)
        self.noptimal_slack = FLOAT(noptimal_slack)

        if niche_elitism not in (None, "selfish", "unselfish"):
            raise ValueError(
                f"'niche_elitism' expected one of (None, 'selfish', 'unselfish'). Received: {niche_elitism}"
            )
        self.niche_elitism = niche_elitism
        self.niche_elitism_int = self.niche_elitism_dict[self.niche_elitism]

        if elite_count == -1 and tourn_count == -1:
            raise ValueError("only 1 of 'elite_count' and 'tourn_count' may be -1")
        elite_count = INT(elite_count) if typing.is_integer(elite_count) else INT(elite_count * pop_size)
        tourn_count = INT(tourn_count) if typing.is_integer(tourn_count) else INT(tourn_count * pop_size)
        elite_count = pop_size - tourn_count if elite_count == -1 else elite_count
        tourn_count = pop_size - elite_count if tourn_count == -1 else tourn_count
        if elite_count + tourn_count > pop_size:
            raise ValueError("'elite_count' + 'tourn_count' should be <= 'pop_size'")
        self.elite_count = INT(elite_count)
        self.tourn_count = INT(tourn_count)

        self.hyperparameters_set = True

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
        disp_rate = INT(disp_rate)

        termination_handler = self.configure_termination(convergence_criteria)

        # Instantiation
        if not self._is_populated:
            if not hasattr(self, "num_niches") or self.num_niches <= 1:
                raise RuntimeError(f"'num_niches' must be greater than one. Got: {self.num_niches}"
                                   ". Call `.add_niches()` first.")
            self.populate()

        # Set parent size
        self.population.resize(
            -1,  # ignore niches
            self.pop_size,  # pop_size
            self.tourn_count + self.elite_count,  # parent_size
        )

        # numba does not like keyword arguments
        self.population.update_hyperparameters(
            self.elite_count,
            self.tourn_count,
            self.tourn_size,
            self.mutation_prob,
            self.mutation_sigma,
            self.crossover_prob,
            self.niche_elitism_int,
            self.noptimal_slack,
            self.violation_factor,
        )

        # Main algorithm loop
        try:
            while not termination_handler(self):
                if disp_rate > 0 and self.current_iter % disp_rate == 0:
                    self._display_progress()

                self.run_iteration()

                if self.logger:
                    self.logger.log_iteration(self.current_iter, self.population)

                # provides hooks for termination control
                self.current_best_obj = self.population.current_optima_obj[0]
                self.mean_fitness = self.population.mean_fitness

                self.current_iter += 1

                if self.callback is not None:
                    self.callback(self.population)

        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Terminating gracefully.")
            pass

        if disp_rate != 0:
            self._display_progress()

    def run_iteration(self):
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

    def evaluate_points(self):
        if self._can_use_fast_path:
            self.evaluator(
                self.population, self.problem.objective, self.problem.fargs
            )
        else:
            self.evaluator(
                self.population, self.problem.objective, self.problem.fargs, self.problem.fkwargs
            )

    def evaluate_and_update_population(self, diversity=True):
        # always called on population initialisation
        self.evaluate_points()
        self.population.evaluate_fitness()  # TODO: provide hooks for termination
        if diversity:
            self.population.evaluate_diversity()  # TODO: provide hooks for termination
        self.population.update_optima()

    def populate(self):
        """
        generate starting population
        """
        if self._is_populated:
            return
        if self.pop_size < 1:
            raise ValueError("'pop_size' must be positive definite.")
        if not self._is_populated:
            # Initialize population
            self.population = Population(
                num_niches=self.num_niches,
                pop_size=self.pop_size,
                ndim=self.problem.ndim,
                rng=self.rng,
                stable_sort=self.stable_sort,
            )
            load_problem_to_population(self.population, self.problem)
            self.population.populate(self.noptimal_slack, self.violation_factor, self.x0)
            self.evaluate_and_update_population(False)

            self._is_populated = True

    def get_results(self) -> dict:
        """
        Returns the final results of the optimization.
        """
        if self.logger:
            self.logger.finalize(self.population)

        if self.population is None:
            raise RuntimeError("Algorithm has not been run yet.")

        return {
            "optima": self.population.current_optima,
            "fitness": self.population.current_optima_fit,
            "objective": self.population.current_optima_obj,
            "penalties": self.population.current_optima_pob - self.population.current_optima_obj,
            "noptimality": self.population.current_optima_nop,
        }

    def _display_progress(self):
        """
        Prints the current progress of the algorithm to the console.
        """
        best_obj = self.population.current_optima_obj
        elapsed = dt.now() - self.start_time
        print(f"Iter: {self.current_iter}. Best Objective: {np.round(best_obj[0], 2)}. Time: {elapsed}")

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
    population: Population,  eval_func: CPUDispatcher, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict
) -> None:
    """
    The fallback "slow path" for non-jitted objectives or
    objectives with fargs/fkwargs.
    """
    population.dither()
    population.select_parents()
    population.generate_offspring()
    eval_func(population, objective_func, fargs, fkwargs)
    population.evaluate_fitness()
    population.evaluate_diversity()
    population.update_optima()


def construct_python_eval_func(  # noqa: C901
        vectorized: bool,
        constraints: bool,
) -> Callable:
    """
    Constructs the appropriate evaluation function based on problem properties.
    """
    if vectorized and constraints:
        def _evaluator(population: Population, objective_func: Callable, fargs: tuple, fkwargs: dict):
            for i in range(population.num_niches):
                population.objective_values[i, :], population.violations[i, :] = objective_func(
                    population.points[i], *fargs, **fkwargs
                )
                population.penalized_objectives[:] = (
                    population.objective_values + population.violations * population.violation_factor
                )

    elif vectorized and not constraints:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            for i in prange(population.num_niches):
                population.objective_values[i, :] = objective_func(population.points[i], *fargs, **fkwargs)
            population.violations[:] = 0.0  # No constraint penalties
            population.penalized_objectives[:] = population.objective_values  # No penalty

    elif not vectorized and constraints:
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.points[i].shape[0]):
                    population.objective_values[i, j], population.violations[i, j] = objective_func(
                        population.points[i, j], *fargs, **fkwargs
                    )
                    population.penalized_objectives[i, j] = (
                        population.objective_values[i, j] + population.violations[i, j] * population.violation_factor
                    )

    else:  # not vectorized and not constraints
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple, fkwargs: dict):
            """
            Evaluates a scalar objective function that returns only obj_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.points[i].shape[0]):
                    population.objective_values[i, j] = objective_func(
                        population.points[i, j], *fargs, **fkwargs
                    )
            population.violations[:] = 0.0  # No constraints
            population.penalized_objectives[:] = population.objective_values  # No penalty

    return _evaluator


# --- JITTED iteration loops ---
@njit
def _njit_iteration_loop(population: Population, eval_func: CPUDispatcher, objective_func: CPUDispatcher, fargs: tuple):
    population.dither()
    population.select_parents()
    population.generate_offspring()
    eval_func(population, objective_func, fargs)
    population.evaluate_fitness()
    population.evaluate_diversity()
    population.update_optima()


def construct_njit_eval_func(  # noqa: C901
        vectorized: bool,
        constraints: bool,
        parallelize: bool,
) -> CPUDispatcher:
    """
    Constructs the appropriate evaluation function based on problem properties.
    """
    if vectorized and constraints:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns obj_vals, violation_vals.
            """
            for i in prange(population.num_niches):
                population.objective_values[i, :], population.violations[i, :] = objective_func(
                    population.points[i], *fargs
                )
            population.penalized_objectives[:] = (
                population.objective_values + population.violations * population.violation_factor
            )

    elif vectorized and not constraints:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a vectorized objective function that returns only obj_vals.
            """
            for i in prange(population.num_niches):
                population.objective_values[i, :] = objective_func(population.points[i], *fargs)
            population.violations[:] = 0.0  # No constraint penalties
            population.penalized_objectives[:] = population.objective_values  # No penalty

    elif not vectorized and constraints:
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns obj_val, violation_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.points[i].shape[0]):
                    population.objective_values[i, j], population.violations[i, j] = objective_func(
                        population.points[i, j], *fargs
                    )
                    population.penalized_objectives[i, j] = (
                        population.objective_values[i, j] + population.violations[i, j] * population.violation_factor
                    )

    else:  # not vectorized and not constraints
        @njit(parallel=parallelize)
        def _evaluator(population: Population, objective_func: CPUDispatcher, fargs: tuple):
            """
            Evaluates a scalar objective function that returns only obj_val.
            """
            for i in prange(population.num_niches):
                for j in range(population.points[i].shape[0]):
                    population.objective_values[i, j] = objective_func(
                        population.points[i, j], *fargs
                    )
            population.violations[:] = 0.0  # No constraints
            population.penalized_objectives[:] = population.objective_values  # No penalty

    return _evaluator
