import numpy as np
from numba import njit
from datetime import datetime as dt
from datetime import timedelta as td
from time import perf_counter
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from mga.commons.types import DEFAULTS
DEFAULTS.update_precision(64)

from mga.problem_definition import MultiObjectiveProblem, OptimizationProblem
from mga.mhmga import MGAProblem
from mga.mhmoo import MOProblem
from mga.utils import plotting, profiling
from mga.utils import termination as term

@njit
def MOObjective(values_array): 
    """ Vectorized = True """
    z = 2 + np.zeros((values_array.shape[0], 3), np.float64)
    for i in range(values_array.shape[0]):
        for j in range(values_array.shape[1]):
            z[i, 0] += np.sin(19 * np.pi * values_array[i, j]) + values_array[i, j] / 1.7
        z[i, 1] = values_array[i, 0]
        z[i, 2] = values_array[i, 1]
    return z
    
@njit
def feasibility_wrapper(values_array):
    objective = MOObjective(values_array)
    feasibility = np.ones(objective.shape, np.bool_)
    return objective, feasibility

@njit
def Objective(values_array): 
    """ Vectorized = True """
    # For the first test, ndim=2 and works for a function with two decision variables
    z = 2 + np.zeros(values_array.shape[0], np.float64)
    for i in range(values_array.shape[0]):
        for j in range(values_array.shape[1]):
            z[i] += np.sin(19 * np.pi * values_array[i, j]) + values_array[i, j] / 1.7
    return z   

lb = 0*np.ones(2)
ub = 1*np.ones(2)

def run_optimize(hyperparameters, timeout, seed):
    problem = OptimizationProblem(
        Objective,
        bounds = (lb, ub),
        maximize = True,
        vectorized = True,
        constraints = False,
        fargs = (),
        fkwargs = {},
        )

    algorithm = MGAProblem(problem, None, None, 0, seed)
    algorithm.add_niches(20)
    algorithm.step(
        **hyperparameters,
        tourn_count=-1, 
        noptimal_slack=1.12,
        disp_rate=0,
        convergence_criteria = term.MultiConvergence(
            [
                term.Timeout(
                    timeout, 
                    start_attribute="start_time"
                ), # timeout
                term.MultiConvergence(
                    [
                        term.FixedValue(
                            5.145, 
                            maximize=True,
                            attribute="current_best_obj"
                        ), # should reach global optimum
                        term.GradientStagnation(
                            window = 50, # implies niter >= 50
                            improvement = 0.01 / 50, # improves less than 0.01 in 50 its
                            maximize = True, 
                            attribute = "mean_fitness",
                        ), # mean fitness stops improving
                    ], 
                    how = "and",
                ), 
            ],
            how = "or",
        ),
    )
    return algorithm.population.shannon, algorithm.population.current_optima_nop.sum()

# def Optimize(x, best_time, n_repeat=3):
def Optimize(x, n_repeat=3):
    global best_time
    hyperparameters = dict(zip(
        ("max_iter", "pop_size", "elite_count", "tourn_size", 
         "mutation_prob", "mutation_sigma", "crossover_prob", "niche_elitism"), x))
    for k in ("max_iter", "pop_size", "tourn_size"):
        hyperparameters[k] = int(hyperparameters[k])
    if hyperparameters["tourn_size"] >= hyperparameters["pop_size"]:
        return np.full(3, np.inf), np.zeros(3, bool)

    hyperparameters["niche_elitism"] = {0:None, 
                                        1:"selfish", 
                                        2:"unselfish"}[hyperparameters["niche_elitism"]]
    try: 
        timeout = 10 * best_time
    except OverflowError:
        timeout = td.max
    
    shannon, nnopt = 0, 0
    time = 0
    for seed in range(1, n_repeat+1):
        start = perf_counter()
        
        sha, n = run_optimize(
            hyperparameters = hyperparameters, 
            timeout = timeout, 
            seed=seed, 
            )
        shannon += sha
        nnopt += n
        
        time += perf_counter() - start
    shannon /= n_repeat
    nnopt /= n_repeat
    time = td(seconds=time/n_repeat)

    best_time = min(best_time, time)

    return np.array([time.total_seconds(), shannon, nnopt]), np.ones(3, bool)
    
# def OptimizeParallelWrapper(xs, n_repeat=3):
#     global best_time
#     ncpus = cpu_count() // 2
#     with Pool(processes=min(len(xs), ncpus)) as process_pool:
#         result = process_pool.starmap(Optimize, [(x, best_time) for x in xs])

#     objectives = np.stack([res[0] for res in result])
#     feasibility = np.stack([res[1] for res in result])

#     for time in objectives[:, 0]:
#         if time < np.inf:
#             best_time = min(best_time, td(seconds=time))
#     return objectives, feasibility

def main(calc = True, plot=True):
    if calc:
        problem = MultiObjectiveProblem(
            Optimize,
            bounds=(
                np.array([50,    10,    0.0, 2,  0.0, 0.0, 0.0, 0]), 
                np.array([10000, 10000, 1.0, 10, 1.0, 2.0, 1.0, 2]), 
                ),
            n_objs=3,
            maximize=np.array([False, True, True]),
            vectorized=False,
            feasibility=True,
            integrality=np.array([True, True, False, True, False, False, False, True]),
        )

        algorithm = MOProblem(
            problem=problem,
            x0 = np.array([200, 25, 0.2, 2, 0.5, 0.3, 0.3, 1])
        )

        algorithm.step(
            max_iter=2,
            pop_size=5,
            pareto_size=200,
            elite_count=0.2,
            tourn_count=-1,
            tourn_size=2,
            mutation_prob=0.5,
            mutation_sigma=0.2,
            crossover_prob=0.3,
            disp_rate=1,
            )
        algorithm.step(
            max_iter=50,
            pop_size=200,
            pareto_size=500,
            elite_count=0.2,
            tourn_count=-1,
            tourn_size=2,
            mutation_prob=0.5,
            mutation_sigma=0.2,
            crossover_prob=0.3,
            disp_rate=1,
            )

        results = algorithm.get_results()
        pareto_points = results["pareto"]
        pareto_objectives = results["objectives"]

        pd.DataFrame(pareto_points).to_csv("logs/pareto_points.csv", index=False, header=False)
        pd.DataFrame(pareto_objectives).to_csv("logs/pareto_objectives.csv", index=False, header=False)
    
    if plot: 
        
        # raise KeyboardInterrupt
        pareto_points = pd.read_csv("logs/pareto_points.csv", header=None).to_numpy()
        pareto_objectives = pd.read_csv("logs/pareto_objectives.csv", header=None).to_numpy()

        
        
        objectives = ["time", "shannon", "n_noptima"]
        for i in range(3):
            fig, ax = plt.subplots()
            ax.scatter(pareto_objectives[:, i-1], pareto_objectives[:, i])
            ax.set_xlabel(f"{objectives[i-1]}")
            ax.set_ylabel(f"{objectives[i]}")

        def zero_safe_divide(num, denom, ret):
            retarr = np.empty_like(num)
            for i in range(retarr.shape[0]):
                for j in range(retarr.shape[1]):
                    if denom[j] != 0:
                        retarr[i, j] = num[i, j] / denom[j]
                    else: 
                        retarr[i, j] = ret
            return retarr
        
        def normalise(array, lb, ub):
            lb = array.min(axis=0)
            ub = array.max(axis=0)
            return zero_safe_divide(array - lb, ub -lb, 0)

        def generate_ticklabels(array, nticks, lb, ub):
            lb = array.min(axis=0)
            ub = array.max(axis=1)
            
            ticks_norm = np.linspace(0, 1, 5)
            
            ticks = []
            for i in range(array.shape[1]):
                ticks.append([f"{val:.2f}" for val in np.linspace(lb[i], ub[i], nticks)])
            return ticks_norm, ticks
        
        time_mask = pareto_objectives[:, 0] < pareto_objectives[:, 0].min()*2
        pareto_points = pareto_points[time_mask, :]
        pareto_objectives = pareto_objectives[time_mask, :]
        
        lb, ub = pareto_objectives.min(axis=0), pareto_objectives.max(axis=0)
        normal_pareto = normalise(pareto_objectives, lb, ub)
        # print(229)
        cloud = pv.PolyData(normal_pareto)
        # print(231)
        surf = cloud.reconstruct_surface()
        # surf = cloud.delaunay_3d()
        # print(234)
        ticks_norm, ticks = generate_ticklabels(pareto_objectives, 5, lb, ub) # label normalised axes with unnormalised labels
        xticks, yticks, zticks = ticks
        # print(237)
        plotter = pv.Plotter()
        # print(239)
        plotter.add_mesh(surf, show_edges=True)#, cmap="viridis")
        # print(241)
        # plotter.add_points(cloud, color="blue", render_points_as_spheres=True, point_size=10)
        # print(243)
        plotter.show_grid(
            xtitle=objectives[0],
            ytitle=objectives[1],
            ztitle=objectives[2],
            axes_ranges=[i for j in zip(lb, ub) for i in j]
            )
        plotter.view_isometric()
        plotter.show()

if __name__=="__main__":
    best_time = td(seconds=1)

    main(True, True)
