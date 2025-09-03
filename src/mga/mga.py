import numpy as np
from datetime import datetime as dt

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
import mga.utils.termination as term
from mga.utils import type_asserts
from mga.problem_definition import OptimizationProblem
from mga.population import Population
from mga.utils.logger import Logger


class MGAProblem:
    """
    Orchestrates the modelling to generate alternatives algorithm.
    Manages the optimization loop, problem state, and logging.
    """
    def __init__(
        self,
        problem: OptimizationProblem,
        log_dir: str|None = None,
        log_freq: int = 1,
        random_seed: int|None = None
    ):
        """
        Initializes the MGA algorithm
        """
        # sanitize inputs
        if not isinstance(problem, OptimizationProblem): raise TypeError(f"'problem' expected an OptimizationProblem. Received: {type(problem)}")
        if log_dir is not None:
            if not isinstance(log_dir, str): raise TypeError(f"'log_dir' expected a string. Received: {type(log_dir)}")
        if not type_asserts.is_integer(log_freq): raise TypeError(f"'log_freq' expected an integer. Received {type(log_freq)}")
        if log_dir is not None: 
            if log_freq == 0 or log_freq < -1: raise ValueError(f"'log_freq' should be -1 or strictly greater than 0. Received: {log_freq}")
        if random_seed is not None: 
            if not type_asserts.is_integer(random_seed): raise TypeError(f"'random_seed' expects integer or None. Received: {type(random_seed)}")

        # Instantiation
        self.problem = problem
        self.rng = np.random.default_rng(random_seed)
        self.stable_sort = random_seed is not None 
        self.logger = Logger(log_dir, log_freq) if log_dir else None
        
        self.population = None
        self.current_iter = 0
        self.start_time = dt.now()

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
        
    def add_niches(self, num_niches:int):
        """
        Adds niches to the problem 
        """
        if not type_asserts.is_integer(num_niches):
            raise TypeError(f"'num_niches' expected an integer. Received: {type(num_niches)}")
        if num_niches < 1: 
            raise ValueError(f"'num_niches' must be positive definite. Received {num_niches}")
        if not self._is_populated:
            self.num_niches = num_niches
        else: 
            self.population.add_niches(num_niches)
            self.num_niches += num_niches

    def step(
        self,
        max_iter: int,
        pop_size: int = 100,
        elite_count: int | float = 0.2,
        tourn_count: int | float = 0.8,
        tourn_size: int = 2,
        mutation_prob: float | tuple[float, float] = 0.3,
        mutation_sigma: float | tuple[float, float] = 0.05,
        crossover_prob: float = 0.4,
        violation_factor: float = 1.0,
        noptimal_slack: float = np.inf,
        niche_elitism: str = "selfish",
        disp_rate: int = 0,
        convergence_criteria: None|term.Convergence|list[term.Convergence] = None,
    ):
        """
        Executes the main optimization loop.
        """
        # sanitise input
        if not isinstance(max_iter, int): 
            raise TypeError(f"'max_iter' expected an int. Received: {type(max_iter)}")
        if max_iter < 1: 
            raise ValueError(f"'max_iter' must be a strictly positive integer. Received: {max_iter}")
        max_iter = INT(max_iter)
        if not isinstance(pop_size, int): 
            raise TypeError(f"'pop_size' expected an int. Received: {type(pop_size)}")
        if pop_size < 1: 
            raise ValueError(f"'pop_size' must be a strictly positive integer. Received {pop_size}")
        pop_size = INT(pop_size)
        if type_asserts.is_float(elite_count): 
            if not 0 <= elite_count <= 1: 
                raise ValueError(f"float 'elite_count' should be in range [0, 1]. Received: {elite_count}")
        elif type_asserts.is_integer(elite_count):
            if not (-1 == elite_count or elite_count >= 0): 
                raise ValueError(f"integer 'elite_count' should be -1 or >= 0. Received: {elite_count}")
        else: 
            raise TypeError(f"'elite_count' expected an int or float. Received: {type(elite_count)}")
        if type_asserts.is_float(tourn_count): 
            if not 0 <= tourn_count <= 1: 
                raise ValueError(f"float 'tourn_count' should be in range [0, 1]. Received: {tourn_count}")
        elif type_asserts.is_integer(tourn_count):
            if not (-1 == tourn_count or tourn_count >= 0): 
                raise ValueError(f"integer 'tourn_count' should be -1 or >= 0. Received: {tourn_count}")
        else: 
            raise TypeError(f"'tourn_count' expected an int or float. Received: {type(tourn_count)}")
        if not type_asserts.is_integer(tourn_size): 
            raise TypeError(f"'tourn_size' expected an integer. Received: {type(tourn_size)}")
        if tourn_size < 1: 
            raise ValueError("'tourn_size' must be a strictly greater than 1.")
        tourn_size = INT(tourn_size)
        if type_asserts.is_float(mutation_prob):
            if not 0 <= mutation_prob <= 1: 
                raise ValueError(f"'mutation_prob' tuple must be in range [0, 1]. Received: {mutation_prob}")
        elif type_asserts.is_array_like(mutation_prob):
            if not type_asserts.array_dtype_is(mutation_prob, "float"): 
                raise TypeError("'mutation_prob' expected float dtype")
            for i, elem in enumerate(mutation_prob):
                if not 0 <= elem <= 1: 
                    raise ValueError(f"elements in 'mutation_prob' must be in range[0, 1]. Received: {elem} at position {i}")
            if len(mutation_prob) == 1: mutation_prob = mutation_prob[0]
            elif len(mutation_prob) != 2: 
                raise ValueError(f"'mutation_prob' should be scalar or of length 2. Received length: {len(mutation_prob)}")
        mutation_prob = FLOAT(mutation_prob)
        if type_asserts.is_float(mutation_sigma):
            pass
        elif type_asserts.is_array_like(mutation_sigma):
            if not type_asserts.array_dtype_is(mutation_sigma, "float"): 
                raise TypeError("'mutation_prob' expected float dtype")
            if len(mutation_sigma) == 1: mutation_sigma = mutation_sigma[0]
            elif len(mutation_sigma) != 2: 
                raise ValueError(f"'mutation_sigma' should be scalar or of length 2. Received length: {len(mutation_sigma)}")
        mutation_sigma = FLOAT(mutation_sigma)
        if type_asserts.is_float(crossover_prob):
            if not 0 <= crossover_prob <= 1: 
                raise ValueError(f"'crossover_prob' tuple must be in range [0, 1]. Received: {crossover_prob}")
        elif type_asserts.is_array_like(mutation_prob):
            if not type_asserts.array_dtype_is(crossover_prob, "float"): 
                raise TypeError("'crossover_prob' expected float dtype")
            for i, elem in enumerate(crossover_prob):
                if not 0 <= elem <= 1: 
                    raise ValueError(f"elements in 'crossover_prob' must be in range[0, 1]. Received: {elem} at position {i}")
            if len(crossover_prob) == 1: crossover_prob = crossover_prob[0]
            elif len() != 2: 
                raise ValueError(f"'crossover_prob' should be scalar or of length 2. Received length: {len(crossover_prob)}")
        crossover_prob = FLOAT(crossover_prob)
        if not type_asserts.is_float(violation_factor):
            raise TypeError(f"'violation_factor' expected a float. Received: {type(violation_factor)}")
        if violation_factor < 0:
            raise ValueError("'violation_factor' should not be negative (sign is handled automagically for minimization/maximization).")
        violation_factor = FLOAT(violation_factor)
        if not type_asserts.is_float(noptimal_slack):
            raise TypeError(f"'noptimal_slack' expected a float. Received: {type(noptimal_slack)}")
        if noptimal_slack < 0: 
            raise ValueError(f"'noptimal_slack' cannot be negative. Received: {noptimal_slack}")
        noptimal_slack = FLOAT(noptimal_slack)
        if not niche_elitism in (None, "selfish", "unselfish"): 
            raise ValueError(f"'niche_elitism' expected one of (`None`, 'selfish', 'unselfish'). Received: {niche_elitism}")
        if not type_asserts.is_integer(disp_rate): 
            raise TypeError(f"'disp_rate' expected an int. Received: {type(disp_rate)}")
        if disp_rate < 0: 
            raise ValueError(f"'disp_rate' cannot be negative. Received: {disp_rate}")
        disp_rate = INT(disp_rate)
        if convergence_criteria is None: pass
        elif isinstance(convergence_criteria, term.Convergence): pass
        else: 
            if not type_asserts.is_array_like(convergence_criteria): 
                raise TypeError(f"'convergence_criteria' expected None, `Convergence` or a list of `Convergence` objects. Received: {type(convergence_criteria)}")
            if not type_asserts.array_dtype_is(convergence_criteria, term.Convergence):
                TypeError(f"elements of 'convergence_criteria' should have dtype `Convergence`")

        # Instantiation
        if not self._is_populated:
            if not hasattr(self, "num_niches"):
                raise RuntimeError("MGA needs niches > 1. Call `.add_niches()` first.")
            self.populate(pop_size, noptimal_slack, violation_factor)

        if elite_count == -1 and tourn_count == -1:
            raise ValueError("only 1 of 'elite_count' and 'tourn_count' may be -1")
        elite_count = INT(elite_count) if type_asserts.is_integer(elite_count) else INT(elite_count*pop_size)
        tourn_count = INT(tourn_count) if type_asserts.is_integer(tourn_count) else INT(tourn_count*pop_size)
        elite_count = pop_size - tourn_count if elite_count == -1 else elite_count
        tourn_count = pop_size - elite_count if tourn_count == -1 else tourn_count
        if elite_count + tourn_count > pop_size:
            raise ValueError("'elite_count' + 'tourn_count' should be weakly less than 'pop_size'")
        if tourn_size > pop_size:
            raise ValueError("'tourn_size' should be less than 'pop_size'")
        # Set parent size 
        self.population.resize(
            pop_size=pop_size, 
            parent_size=tourn_count+elite_count, 
            stable_sort=self.stable_sort,
            )

        # Setup termination criteria
        if convergence_criteria is None:
            convergence_criteria = []
        termination_handler = term.MultiConvergence(
            criteria=[term.Maxiter(max_iter)] + convergence_criteria
        )

        # Main algorithm loop
        while not termination_handler(self):
            if disp_rate > 0 and self.current_iter % disp_rate == 0:
                self._display_progress()

            self.population.evolve(
                elite_count=elite_count,
                tourn_count=tourn_count,
                tourn_size=tourn_size,
                mutation_prob=mutation_prob,
                mutation_sigma=mutation_sigma,
                crossover_prob=crossover_prob,
                niche_elitism=niche_elitism,
                rng=self.rng,
                stable_sort=self.stable_sort,
            )
            
            self.population.evaluate_and_update(noptimal_slack, violation_factor)

            if self.logger:
                self.logger.log_iteration(self.current_iter, self.population)

            self.current_iter += 1

        print("Termination criteria met.")
        if self.logger:
            self.logger.finalize(self.population)

    def populate(self, pop_size: int, noptimal_slack: float, violation_factor: float):
        """
        generate starting population
        """
        if self._is_populated:
            return
        if pop_size < 1: 
            raise ValueError("'pop_size' must be positive definite.")
        if not self._is_populated:
            # Initialize population
            self.population = Population(
                problem=self.problem,
                num_niches=self.num_niches,
                pop_size=pop_size,
                rng=self.rng,
            )
            self.population.populate()
            self.population.evaluate_and_update(noptimal_slack, violation_factor)
            self._is_populated = True

    def get_results(self) -> dict:
        """
        Returns the final results of the optimization.
        """
        if self.population is None:
            raise RuntimeError("Algorithm has not been run yet.")
        
        return {
            "optima": self.population.current_optima,
            "fitness": self.population.current_optima_fit,
            "objective": self.population.current_optima_obj,
            "noptimality": self.population.current_optima_nop,
        }

    def _display_progress(self):
        """
        Prints the current progress of the algorithm to the console.
        """
        best_obj = self.population.current_optima_obj
        elapsed = dt.now() - self.start_time
        print(f"Iter: {self.current_iter}. Best Objectives: {np.round(best_obj, 2)}. Time: {elapsed}")