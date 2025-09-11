import numpy as np 
from numba import njit, jit
from datetime import datetime as dt
import warnings

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
import mga.utils.termination as term
from mga.utils import type_asserts
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
        self.npareto = INT(0.0)
        self.mutation_prob = FLOAT(0.0)
        self.mutation_sigma = FLOAT(0.0)
        self.crossover_prob = FLOAT(0.0)

    def step(
            self, 
            max_iter: int|float = np.inf, # max no of iterations in this step
            pop_size: int = 100, # max no of individuals in each niche
            npareto: int = None,
            mutation_prob: float|tuple[float, float] = (0.5, 0.75), # mutation probability
            mutation_sigma: float|tuple[float, float] = 0.1, # standard deviation of gaussian noise for mutation (gets scaled)
            crossover_prob: float|tuple[float, float] = 0.4, # crossover probability
            disp_rate: int = 0,
            convergence_criteria: None|term.Convergence|list[term.Convergence] = None,
            ):
        self.npareto = pop_size if npareto is None else npareto

        # Setup termination criteria
        if convergence_criteria is None:
            convergence_criteria = []
        termination_handler = term.MultiConvergence(
            criteria=[term.Maxiter(max_iter)] + convergence_criteria
        )

        if not self._is_populated:
            self.populate(pop_size)
        mutation_sigma = mutation_sigma*(self.problem.upper_bounds - self.problem.lower_bounds)

        # Main algorithm loop
        while not termination_handler(self):
            if disp_rate > 0 and self.current_iter % disp_rate == 0:
                self._display_progress()

            # TODO: dither mutation/crossover params
            self.population.evolve(
                npareto = self.npareto,
                mutation_prob=mutation_prob,
                mutation_sigma=mutation_sigma,
                crossover_prob=crossover_prob,
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
        print(f"Iter: {self.current_iter}. Pareto front: {self.population.pareto.shape[0]}/{self.npareto}. Time: {elapsed}")
