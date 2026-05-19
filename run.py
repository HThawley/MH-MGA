import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd  # noqa: F401

os.environ["MGA_JIT_ENABLED"] = "0"

from mga.commons.numba_overload import njit  # noqa: E402
from mga.problem_definition import OptimizationProblem  # noqa: E402
from mga.mhmga import MGAProblem  # noqa: E402
from mga.utils import plotting  # noqa: E402


@njit(fastmath=True)
def complex_objective(values_array):
    """
    A complex, multivariate objective function for testing the algorithm.
    This function is designed to be vectorized.
    """
    z = 2 + np.zeros(values_array.shape[0], np.float64)
    for i in range(values_array.shape[0]):
        for j in range(values_array.shape[1]):
            z[i] += np.sin(19 * np.pi * values_array[i, j]) + values_array[i, j] / 1.7
    return z


@njit(fastmath=True)
def convex_objective(values_array):
    """
    2D trig function with a single local minimum.
    """
    z = np.full(values_array.shape[0], 3.15, dtype=np.float64)
    for i in range(values_array.shape[0]):
        for j in range(values_array.shape[1]):
            z[i] += np.cos(3 * (values_array[i, j] - 0.5))
    return z


def run(obj=complex_objective, seed=None, file_prefix="logs/testprob"):
    global algorithm
    # 1. Define the optimization problem
    problem = OptimizationProblem(
        objective=obj,
        bounds=(np.zeros(2), np.ones(2)),
        maximize=True,
        vectorized=True,
        constraints=False,
    )

    # 2. Configure the MGA algorithm
    algorithm = MGAProblem(
        problem=problem,
        x0=np.array([0.5, 0.5]),
        log_dir=file_prefix,
        log_freq=-1,
        random_seed=seed,
        include_obj_in_fitness=True,
        angular_fitness=True,
    )

    algorithm.add_niches(num_niches=20)
    # algorithm.starting_points = np.full((20, 50, 2), 0.5)
    algorithm.update_hyperparameters(
        max_iter=200,
        pop_size=50,
        champ_count=0,
        elite_count=0.2,
        tourn_count=-1,
        tourn_size=3,
        mutation_prob=0.25,
        mutation_sigma=0.01,
        mutation_alpha=0.05,
        crossover_prob=0.0,
        niche_elitism="selfish",
        noptimal_rel=0.12,
        objective_scaler=16.18)
    # algorithm.set_verbosity(disp_rate=1, verbose=2)
    algorithm.step()

    # algorithm.add_niches(num_niches=10)
    # algorithm.update_hyperparameters(
    #     max_iter=50,
    #     pop_size=200,
    #     champ_count=5,
    #     elite_count=0.2,
    #     tourn_count=-1,
    #     tourn_size=2,
    #     mutation_prob=0.25,
    #     mutation_sigma=(0.05, 0.5),
    #     crossover_prob=0.0,
    #     niche_elitism="selfish",
    #     noptimal_rel=0.12,
    #     space_scaler=np.array([2.0, 1.0]),
    # )
    # algorithm.step()

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
    #     noptimal_rel=0.12,
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
    return algorithm


def plot(file_prefix=None, obj=None):
    print("\nGenerating plots...")
    if file_prefix is None:
        if obj is None:
            plotting.plot_landscape(
                complex_objective,
                convex_objective,
                np.zeros(2),
                np.ones(2),
                1 - 0.12,
                "original",
                "convex"
            )
        else:
            plotting._landscape(
                obj,
                np.zeros(2),
                np.ones(2),
                1 - 0.12,
                cbar=True
            )
    else:
        if obj is None:
            plotting.plot_noptima(file_prefix)
        else:
            ax, _ = plotting._landscape(
                obj,
                np.zeros(2),
                np.ones(2),
                1 - 0.12,
                cbar=True
            )
            plotting.plot_noptima(file_prefix, ax)

        plotting.plot_stat_evolution(file_prefix)
        plotting.plot_vesa(file_prefix)
        plotting.plot_shannon(file_prefix)

    plotting.show()


def inspect_recomb(obj=complex_objective, points=None, random_seed=1, **hyperparameters):
    if points is None:
        points = np.meshgrid(
            np.arange(0.8, 1.0, 0.02),
            np.arange(0.8, 1.0, 0.02),
        )
        points = np.stack((points[0].flatten(), points[1].flatten())).T

    problem = OptimizationProblem(
        objective=obj,
        bounds=(np.zeros(2), np.ones(2)),
        maximize=True,
        vectorized=True,
        constraints=False,
    )

    algorithm = MGAProblem(
        problem=problem,
        x0=None,
        starting_points=points,
        log_dir=None,
        log_freq=-1,
        random_seed=random_seed,
    )

    if points.ndim > 1:
        hyperparameters["pop_size"] = points.shape[0]

    algorithm.add_niches(num_niches=1)
    algorithm.update_hyperparameters(**hyperparameters)
    result = algorithm.inspect_recombination(points)

    fig, (ax1, ax2, cax) = plt.subplots(
        1, 3, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 1, 0.05]}
    )
    vmin = min(result["parents_objectives"].min(), result["offspring_objectives"].min())
    vmax = max(result["parents_objectives"].max(), result["offspring_objectives"].max())
    x1min = min(result["parents_points"][:, 0].min(), result["offspring_points"][:, 0].min())
    x1max = max(result["parents_points"][:, 0].max(), result["offspring_points"][:, 0].max())
    x2min = min(result["parents_points"][:, 1].min(), result["offspring_points"][:, 1].min())
    x2max = max(result["parents_points"][:, 1].max(), result["offspring_points"][:, 1].max())

    def _plot(ax, name):
        sc = ax.scatter(
            result[f"{name}_points"][:, 0],
            result[f"{name}_points"][:, 1],
            c=result[f"{name}_objectives"],
            vmin=vmin,
            vmax=vmax,
            cmap='viridis',
            edgecolor='k'
        )
        ax.set_title(name.capitalize())
        ax.set_xlim(x1min - 0.01, x1max + 0.01)
        ax.set_ylim(x2min - 0.01, x2max + 0.01)
        return sc

    sc1 = _plot(ax1, "parents")
    _plot(ax2, "offspring")

    fig.colorbar(sc1, cax=cax, label='Objective Value')
    return result


if __name__ == "__main__":

    plot()

    file_prefix = "logs/testprob"
    algorithm = run(convex_objective, seed=2, file_prefix=file_prefix)
    plot(file_prefix, convex_objective)

    # for i in range(20):
    #     algorithm.step()
    #     results = algorithm.get_results()
    #     plot(file_prefix)

    #     df = pd.read_csv('logs/testprob-noptima.csv')
    #     df = df.drop(index=df.index)
    #     df.to_csv('logs/testprob-noptima.csv', index=False, header=True)

    # points = algorithm.population.points[0, :100, :].copy()

    # points = np.array([0.5, 0.5])

    # hyperparameters = dict(
    #     max_iter=1,
    #     pop_size=50,
    #     champ_count=0,
    #     elite_count=0.2,
    #     tourn_count=-1,
    #     tourn_size=3,
    #     mutation_prob=0.25,
    #     mutation_sigma=0.005,
    #     mutation_alpha=0.05,
    #     crossover_prob=0.0,
    #     niche_elitism="selfish",
    #     noptimal_rel=0.12,
    # )

    # results = inspect_recomb(points, random_seed=1, **hyperparameters)
