import numpy as np
from numba import njit

from mga.problem_definition import OptimizationProblem
from mga.mhmga import MGAProblem
from mga.utils import plotting


@njit(fastmath=True)
def objective_function(values_array):
    """
    A complex, multivariate objective function for testing the algorithm.
    This function is designed to be vectorized.
    """
    z = 2 + np.zeros(values_array.shape[0], np.float64)
    for i in range(values_array.shape[0]):
        for j in range(values_array.shape[1]):
            z[i] += np.sin(19 * np.pi * values_array[i, j]) + values_array[i, j] / 1.7
    return 10 - z


def main(run=True, plot=True, seed=None):
    """
    Configures and runs the MGA algorithm.
    """
    global algorithm
    FILE_PREFIX = "logs/testprob"
    # FILE_PREFIX = None
    if run:
        # 1. Define the optimization problem
        problem = OptimizationProblem(
            objective=objective_function,
            bounds=(np.zeros(2), np.ones(2)),
            maximize=False,
            vectorized=True,
            constraints=False,
        )

        # 2. Configure the MGA algorithm
        algorithm = MGAProblem(
            problem=problem,
            x0=None,
            log_dir=FILE_PREFIX,
            log_freq=-1,
            random_seed=seed,
        )

        algorithm.add_niches(num_niches=20)
        algorithm.update_hyperparameters(
            max_iter=200,
            pop_size=20,
            elite_count=0.2,
            tourn_count=-1,
            tourn_size=2,
            mutation_prob=0.25,
            mutation_sigma=(0.05, 0.5),
            crossover_prob=0.0,
            niche_elitism=None,  # "selfish",
            noptimal_slack=1.12,
        )
        algorithm.step(disp_rate=1)

        # algorithm.update_hyperparameters(
        #     max_iter=200,
        #     pop_size=20,
        #     elite_count=0,
        #     tourn_count=-1,
        #     tourn_size=2,
        #     mutation_prob=0.3,
        #     mutation_sigma=0.05,
        #     crossover_prob=0.0,
        #     niche_elitism="selfish",
        #     noptimal_slack=1.12,
        # )
        # algorithm.step(disp_rate=5)

        # 4. Terminate and get results
        results = algorithm.get_results()
        print("\n--- Final N-optima ---")
        for i in range(results["optima"].shape[0]):
            print(
                f"Niche {i}: Point={results['optima'][i]}, "
                f"Fitness={results['fitness'][i]:.4f}, "
                f"Objective={results['objective'][i]:.4f}, "
                f"Is N-optimal={results['noptimality'][i]}"
            )

    if plot:
        # 5. Print profiling information and plot results
        # profiling.print_profiler_summary()
        if FILE_PREFIX is not None:
            print("\nGenerating plots...")
            plotting.plot_noptima(FILE_PREFIX)
            plotting.plot_stat_evolution(FILE_PREFIX)
            plotting.plot_vesa(FILE_PREFIX)
            plotting.plot_shannon(FILE_PREFIX)
            plotting.show()
    return algorithm


if __name__ == "__main__":
    algorithm = main(True, True)
