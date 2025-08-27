import numpy as np
from datetime import datetime as dt

import mga.utils.termination as term
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
        log_dir: str = None,
        log_freq: int = 1,
        random_seed: int = None
    ):
        """
        Initializes the MGA algorithm
        """
        self.problem = problem
        self.rng = np.random.default_rng(random_seed)
        self.stable_sort = random_seed is not None
        self.logger = Logger(log_dir, log_freq) if log_dir else None
        
        self.population = None
        self.current_iter = 0
        self.start_time = dt.now()

                # State and hyperparameter storage
        self._is_initialized = False
        self.pop_size = 0
        self.elite_count = 0
        self.tourn_count = 0
        self.tourn_size = 0
        self.mutation_prob = 0.0
        self.mutation_sigma = 0.0
        self.crossover_prob = 0.0
        self.niche_elitism = None
        self.noptimal_slack = 1.0
        
    def add_niche(self, num_niches:int, pop_size:int):
        """
        Initializes the population or adds new niches.

        This method must be called with a non-zero 'num_niches' before
        the '.step()' method can be used.
        """
        if num_niches < 0: 
            raise ValueError("'num_niches' must be positive definite.")
        if pop_size < 1: 
            raise ValueError("'pop_size' must be positive definite.")

        if not self._is_initialized:
            # Initialize population
            self.population = Population(
                problem=self.problem,
                num_niches=num_niches,
                pop_size=pop_size,
                rng=self.rng,
            )
            self.population.initialize()
            self.population.evaluate_and_update(np.inf)
            self._is_initialized = True
        else: 
            self.population.add_niches(num_niches)

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
        noptimal_slack: float = np.inf,
        niche_elitism: str = "selfish",
        disp_rate: int = 0,
        convergence_criteria: list = None
    ):
        """
        Executes the main optimization loop.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "The MGA must be initialized by calling '.add_niche(n, p)' before it can be run."
            )
        
        assert elite_count!=-1 or tourn_count !=-1, "only 1 of 'elite_count' and 'tourn_count' may be -1"
        elite_count = elite_count if isinstance(elite_count, int) else int(elite_count*pop_size)
        tourn_count = tourn_count if isinstance(tourn_count, int) else int(tourn_count*pop_size)
        elite_count = pop_size - tourn_count if elite_count == -1 else elite_count
        tourn_count = pop_size - elite_count if tourn_count == -1 else tourn_count
        assert elite_count + tourn_count <= pop_size, "'elite_count' + 'tourn_count' should be weakly less than 'pop_size'"
        assert pop_size > tourn_size, "'tourn_size' should be less than 'pop_size'"

        # Setup termination criteria
        if convergence_criteria is None:
            convergence_criteria = []
        
        termination_handler = term.MultiConvergence(
            criteria=[term.Maxiter(max_iter)] + convergence_criteria
        )

        self.population.resize(pop_size, tourn_count+elite_count, self.stable_sort)

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
            
            self.population.evaluate_and_update(noptimal_slack)

            if self.logger:
                self.logger.log_iteration(self.current_iter, self.population)

            self.current_iter += 1

        print("Termination criteria met.")
        if self.logger:
            self.logger.finalize(self.population.get_best_solutions())
    
    def get_results(self) -> dict:
        """
        Returns the final results of the optimization.
        """
        if self.population is None:
            raise RuntimeError("Algorithm has not been run yet.")
        
        best_solutions = self.population.get_best_solutions()
        return {
            "noptima": best_solutions['points'],
            "nfitness": best_solutions['fitness'],
            "nobjective": best_solutions['objective'],
            "nnoptimality": best_solutions['is_noptimal'],
        }

    def _display_progress(self):
        """
        Prints the current progress of the algorithm to the console.
        """
        best_obj = self.population.get_best_objective_per_niche()
        elapsed = dt.now() - self.start_time
        print(f"Iter: {self.current_iter}. Best Objectives: {np.round(best_obj, 2)}. Time: {elapsed}")