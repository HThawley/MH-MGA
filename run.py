import numpy as np
from numba import njit

from mga.problem_definition import OptimizationProblem
from mga.mga import MGAProblem
from mga.utils import plotting, profiling

@njit
def objective_function(values_array):
    """
    A complex, multivariate objective function for testing the algorithm.
    This function is designed to be vectorized.
    """
    z = 2 + np.zeros(values_array.shape[0], np.float64)
    for i in range(values_array.shape[0]):
        for j in range(values_array.shape[1]):
            z[i] += np.sin(19 * np.pi * values_array[i, j]) + values_array[i, j] / 1.7
    return z

def main():
    """
    Configures and runs the MGA algorithm.
    """
    # FILE_PREFIX = "logs/testprob-z"
    FILE_PREFIX = None
    
    # 1. Define the optimization problem
    problem = OptimizationProblem(
        objective=objective_function,
        bounds=(np.zeros(2), np.ones(2)),
        maximize=True,
        vectorized=True,
        constraints=False,
    )

    # 2. Configure the MGA algorithm
    algorithm = MGAProblem(
        problem=problem,
        log_dir=FILE_PREFIX,
        log_freq=500,
        random_seed=1, 
    )

    algorithm.add_niches(num_niches=3)
    
    # 3. Run the optimization
    algorithm.step(
        max_iter=200,
        pop_size=100,
        elite_count=0.2,
        tourn_count=-1, # -1 means it will take the remainder of pop_size
        tourn_size=3,
        mutation_prob=0.33,
        mutation_sigma=0.05,
        crossover_prob=0.3,
        niche_elitism=None,#"selfish",
        noptimal_slack=1.12,
        disp_rate=500,
    )

    # 4. Terminate and get results
    results = algorithm.get_results()
    print("\n--- Final N-optima ---")
    for i in range(results['optima'].shape[0]):
        print(f"Niche {i}: Point={results['optima'][i]}, "
              f"Fitness={results['fitness'][i]:.4f}, "
              f"Objective={results['objective'][i]:.4f}, "
              f"Is N-optimal={results['noptimality'][i]}")
    
    # 5. Print profiling information and plot results
    # profiling.print_profiler_summary()
    
    print("\nGenerating plots...")
    if FILE_PREFIX is not None:
        plotting.plot_noptima(FILE_PREFIX)
        plotting.plot_stat_evolution(FILE_PREFIX)
        plotting.plot_vesa(FILE_PREFIX)
        plotting.plot_shannon(FILE_PREFIX)
        plotting.show()

if __name__ == "__main__":
    main()