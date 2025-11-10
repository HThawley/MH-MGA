import numpy as np 
from numba import njit, jit
from datetime import datetime as dt
import warnings

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
import mga.utils.termination as term
from mga.utils import typing
from mga.problem_definition import MultiObjectiveProblem
from mga.moopopulation import Pareto


class MOProblem:
    def __init__(
            self, 
            problem: MultiObjectiveProblem,
            x0: np.ndarray|None = None,
            random_seed: int|None = None,
            ):
        if not isinstance(problem, MultiObjectiveProblem):
            raise TypeError("problem must be an instance of MultiObjectiveProblem")

        # Instantiation
        self.problem = problem
        self.rng = np.random.default_rng(random_seed)
        self.stable_sort = random_seed is not None
        self.x0 = x0

        self.population = None
        self.current_iter = 0
        self.start_time = dt.now()

        # State and hyperparameter storage
        self._is_populated = False
        self.pop_size = INT(0)
        self.pareto_size = INT(0.0)
        self.mutation_prob = FLOAT(0.0)
        self.mutation_sigma = FLOAT(0.0)
        self.crossover_prob = FLOAT(0.0)

    def step(
            self, 
            max_iter: int|float = np.inf, # max no of iterations in this step
            pop_size: int = 100, # max no of individuals in each niche
            pareto_size: int = None,
            elite_count: int | float = 0.2, 
            tourn_count: int | float = 0.8,
            tourn_size: int = 2,
            mutation_prob: float|tuple[float, float] = (0.5, 0.75), # mutation probability
            mutation_sigma: float|tuple[float, float] = 0.1, # standard deviation of gaussian noise for mutation (gets scaled)
            crossover_prob: float|tuple[float, float] = 0.4, # crossover probability
            disp_rate: int = 0,
            convergence_criteria: None|term.Convergence|list[term.Convergence] = None,
            ):
        self.pareto_size = pop_size if pareto_size is None else pareto_size

        if elite_count == -1 and tourn_count == -1:
            raise ValueError("only 1 of 'elite_count' and 'tourn_count' may be -1")
        elite_count = INT(elite_count) if typing.is_integer(elite_count) else INT(elite_count*pop_size)
        tourn_count = INT(tourn_count) if typing.is_integer(tourn_count) else INT(tourn_count*pop_size)
        elite_count = pop_size - tourn_count if elite_count == -1 else elite_count
        tourn_count = pop_size - elite_count if tourn_count == -1 else tourn_count
        if elite_count + tourn_count > pop_size:
            raise ValueError("'elite_count' + 'tourn_count' should be weakly less than 'pop_size'")
        if tourn_size > pop_size:
            raise ValueError("'tourn_size' should be less than 'pop_size'")

        # Setup termination criteria
        if convergence_criteria is None:
            convergence_criteria = []
        termination_handler = term.MultiConvergence(
            criteria=[term.Maxiter(max_iter)] + convergence_criteria
        )

        if not self._is_populated:
            self.populate(pop_size)
        else: 
            self.population.resize(pop_size)

        # Main algorithm loop
        while not termination_handler(self):
            if disp_rate > 0 and self.current_iter % disp_rate == 0:
                self._display_progress()

            sigma = dither(mutation_sigma, self.rng)*(self.problem.upper_bounds - self.problem.lower_bounds)
            self.population.evolve(
                pareto_size=self.pareto_size,
                elite_count=elite_count, 
                tourn_count=tourn_count, 
                tourn_size=tourn_size,
                mutation_prob=dither(mutation_prob, self.rng),
                mutation_sigma=sigma, 
                crossover_prob=dither(crossover_prob, self.rng),
            )
            self.current_iter += 1
        
    def populate(self, pop_size):
        self.pop_size = pop_size
        if self._is_populated:
            return
        if pop_size < 1: 
            raise ValueError("'pop_size' must be positive definite.")
        if not self._is_populated:
            # Initialize population
            self.population = Pareto(
                problem=self.problem,
                pop_size=pop_size,
                rng=self.rng,
                stable_sort=self.stable_sort,
            )
            self.population.populate()
            self._is_populated = True

    def get_results(self) -> dict:
        """
        Returns the final results of the optimization.
        """
        if self.population is None:
            raise RuntimeError("Algorithm has not been run yet.")
        
        return {
            "pareto": self.population.pareto,
            "objectives": self.population.pareto_objs,
        }

    def _display_progress(self):
        """
        Prints the current progress of the algorithm to the console.
        """
        elapsed = dt.now() - self.start_time
        print(f"Iter: {self.current_iter}. Pareto front: {self.population.pareto.shape[0]}/{self.pareto_size}. Time: {elapsed}")


def dither(parameter, rng):
    if hasattr(parameter, "__iter__"):
        return _dither(*parameter, rng)
    else:
        return parameter

@njit
def _dither(lower, upper, rng):
    return rng.uniform(lower, upper)