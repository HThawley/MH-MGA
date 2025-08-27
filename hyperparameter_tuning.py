# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:43:32 2025

@author: u6942852
"""
import numpy as np
from numba import njit
from collections.abc import Callable, Sequence
from functools import partial 
from datetime import datetime as dt
from datetime import timedelta as td
from time import perf_counter

import termination_criteria as tc

def draw_bool(rng):
    return rng.integers(2, dtype=bool)

def draw_float(lb, ub, rng):
    return (rng.random() * (ub-lb)) + lb

def draw_int(lb, ub, rng):
    return rng.integers(lb, ub)

def draw_dither_float(lb, ub, rng):
    a = draw_float(lb, ub, rng)
    b = draw_float(lb, ub, rng)
    if a > b: 
        return (b, a)
    elif a < b:
        return (a, b)
    else: # equal try again (unlikely)
        return draw_dither_float(lb, ub, rng)
    
def draw_dither_int(lb, ub, rng):
    a = draw_int(lb, ub, rng)
    b = draw_int(lb, ub, rng)
    if a > b: 
        return (b, a)
    elif a < b:
        return (a, b)
    else: 
        # Careful: potential for very long or infinite loop if ub-lb is small
        # return draw_dither_int(lb, ub, rng)
        return (a, b)
    
def Draw(dtype, rng, lb=None, ub=None, dither=False):
    if dither == "optional":
        dither = draw_bool(rng)
    if dtype is bool:
        return draw_bool(rng)
    elif dtype is int:
        if dither: 
            return draw_dither_int(lb, ub, rng)
        else: 
            return draw_int(lb, ub, rng)
    if dtype is float: 
        if dither: 
            return draw_dither_float(lb, ub, rng)
        else: 
            return draw_float(lb, ub, rng)

class Hyperparameter:
    def __init__(self, name, dtype, lb, ub, dither=False):
        assert dtype in (float, int, bool), "Dtype: {dtype} not supported"
        if dtype in (float, int):
            assert lb is not None, f"Hyperparameter {name}. Bounds must be provided for {dtype} hyperparameters"
            assert lb is not None, f"Hyperparameter {name}. Bounds must be provided for {dtype} hyperparameters"
            assert ub > lb, f"Hyperparameter {name}. Upper bound must be strictly larger than lower bound"
        assert dither in (False, True, "optional")
        if dtype is bool:
            assert dither is False
        self.name = name
        self.dtype = dtype
        self.lb = lb
        self.ub = ub
        self.dither = dither

class HyperparameterMC:
    def __init__(
            self,
            func,
            maximize: bool = False, 
            best_n: int = 1,
            ):
        self.func = func
        self.maximize = maximize
        
        self.parameters: list[str] = []
        self.values: dict[str, Hyperparameter] = {}
        self.interd: list[str] = []
        
        self.best_n = best_n
        self.sense = 1.0 if self.maximize else -1.0
        self.zero = None
        
        self.best_score = [None] * self.best_n
        self.best_set = [{}] * self.best_n
    
    def AddParameter(self, name, dtype, lb=None, ub=None, dither=False):
        param = Hyperparameter(name, dtype, lb, ub, dither)
        setattr(self, name, param)
        self.parameters.append(name)
        
    def _interdependence(self, 
                         names: list[str], 
                         target: int|float|str, 
                         operation = sum, 
                         modify: list[str]|str = "all", 
                         how: str = "scale", 
                         sense: str = "lt",
                         ):
        
        value_total = operation([self.values[name] for name in names])
        if isinstance(target, str):
            target = self.values[names]
            
        if "g" in sense: 
            if value_total > target: 
                return 
        elif "e" in sense:
            if value_total == target:
                return
        else: # "l" in sense
            if value_total < target:
                return 
        
        nmod = len(modify)
        if how == "scale":
            adj = value_total / target
            for name in modify:
                self.values[name] *= adj
        else: # how == "clip"
            adj = value_total - target
            adj /= nmod
            for name in modify:
                self.values[name] -= adj
        return
    
    def AddInterdependence(
            self, 
            names: list[str], 
            target: int|float|str, 
            operation = sum, 
            modify: list[str]|str = "all", 
            how: str = "clip", 
            sense: str = "lt"
            ):
        if modify == "all":
            modify = names
        if isinstance(modify, str):
            assert modify in names
            modify = [modify]
        if isinstance(modify, list):
            assert all([m in names for m in modify])
        assert how in ("scale", "clip")
        assert sense in ("gt", "ge", "eq", "le", "lt")
        
        func = partial(self._interdependence,
            names, target, operation, modify, how, sense)
        setattr(self, f"interd{len(self.interd)}", func)
        self.interd.append(f"interd{len(self.interd)}")
        
    def DrawParameters(self):
        rng = np.random.default_rng()
        for name in self.parameters:
            hp = getattr(self, name)
            self.values[name] = Draw(hp.dtype, rng, hp.lb, hp.ub, hp.dither)
        for interd in self.interd:
            getattr(self, interd)()
            
    def RunObj(self):
        score = self.func(self)
        if self.zero is None:
            if isinstance(score, td):
                self.zero = td(0)
            elif isinstance(score, (float, int)):
                self.zero = 0.0
            else: 
                raise Exception*"Score is unknown type"
        
        for n in range(self.best_n):
            if self.best_score[n] is None:
                self.UpdateBest(score, n)
                break
            delta = self.sense * (score - self.best_score[n])
            if delta > self.zero:
                self.UpdateBest(score, n)
                break

    def UpdateBest(self, score, n):
        self.best_score.insert(n, score)
        self.best_score.pop()
        self.best_set.insert(n, self.values)
        self.best_set.pop()


#%% Execution
if __name__=="__main__":

    best_time = td.max

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

    def runOptimize(hyperparameters, timeout, seed):
        problem = Problem(
            Objective,
            bounds = (lb, ub),
            x0 = "uniform", 
            maximize = True,
            vectorized = True,
            constraint_violation=False,
            fargs = (),
            fkwargs = {},
            random_seed = seed,
            )

        problem.Step(
            **hyperparameters, 
            tournk = -1,
            new_niche=25,
            slack=1.12, 
            disp_rate=0,
            convergence = tc.MultiConvergence(
                [
                    tc.Timeout(
                        timeout, 
                        start_attribute="start_time"
                    ), # timeout
                    tc.MultiConvergence(
                        [
                            tc.FixedValue(
                                5.145, 
                                maximize=True,
                                attribute="optimal_obj"
                            ), # should reach global optimum
                            tc.GradientStagnation(
                                window = 50, # implies niter >= 50
                                improvement = 0.01 / 50, # improves less than 0.01 in 50 its
                                maximize = True, 
                                attribute = "_mean_fit",
                            ), # mean fitness stops improving
                        ], 
                        how = "and",
                    ), 
                ],
                how = "or",
            ),
        )
        return problem._shannon, problem.nnoptimality.sum()
        

    def Optimize(x, n_repeat=3):
        global best_time
        hyperparameters = dict(zip(
            ("maxiter", "popsize", "elitek", "tournsize", "mutation", "sigma", "crossover", "niche_elitism"), x))
        for k in ("maxiter", "popsize", "tournsize"):
            hyperparameters[k] = int(hyperparameters[k])
        if hyperparameters["tournsize"] >= hyperparameters["popsize"]:
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
            
            sha, n = runOptimize(
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
        if time < best_time:
            best_time = time

        return np.array([time.total_seconds(), shannon, nnopt]), np.ones(3, bool)
        # return time.total_seconds(), shannon_index, vesa


    
    from control import Problem
    from mhmoo import MOProblem
    
    problem = MOProblem(
        func = Optimize,
        bounds = (
            np.array([50,    10,    0.0, 2,  0.0, 0.0, 0.0, 0]), 
            np.array([10000, 10000, 1.0, 10, 1.0, 2.0, 1.0, 2]), 
            ),
        n_objs = 3,
        integrality = np.array([True, True, False, True, False, False, False, True]),
        feasibility = True, 
        maximize = [False, True, True], 
        vectorized = False,
    )

    problem.Step(
        npareto = 200,
        maxiter = 30, 
        popsize = 200,
        mutation = 0.5,
        sigma = 0.1,
        crossover = 0.3, 
        disp_rate =  1,
    )

    pareto_points, pareto_objectives = problem.Terminate()
    
    import pandas as pd 
    pd.DataFrame(pareto_points).to_csv("pareto_points.csv", index=False, header=False)
    pd.DataFrame(pareto_objectives).to_csv("pareto_objectives.csv", index=False, header=False)
    # raise KeyboardInterrupt
    import pandas as pd 
    pareto_points = pd.read_csv("pareto_points.csv", header=None).to_numpy()
    pareto_objectives = pd.read_csv("pareto_objectives.csv", header=None).to_numpy()

    #%% 
    
    objectives = ["time", "shannon", "n_noptima"]
    import matplotlib.pyplot as plt
    for i in range(3):
        fig, ax = plt.subplots()
        ax.scatter(pareto_objectives[:, i-1], pareto_objectives[:, i])
        ax.set_xlabel(f"{objectives[i-1]}")
        ax.set_ylabel(f"{objectives[i]}")

    #%%
    import pyvista as pv
    import numpy as np
    
    def normalise(array):
        lb = array.min(axis=0)
        ub = array.max(axis=0)
        return (array - lb) / (ub -lb)

    def generate_ticklabels(array, nticks):
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
    normal_pareto = normalise(pareto_objectives)
    cloud = pv.PolyData(normal_pareto)
    surf = cloud.reconstruct_surface()
    # surf = cloud.delaunay_3d()
    
    ticks_norm, ticks = generate_ticklabels(pareto_objectives, 5) # label normalised axes with unnormalised labels
    xticks, yticks, zticks = ticks
    
    plotter = pv.Plotter()
    
    plotter.add_mesh(surf, show_edges=True)#, cmap="viridis")
    
    # plotter.add_points(cloud, color="blue", render_points_as_spheres=True, point_size=10)

    plotter.show_grid(
        xtitle=objectives[0],
        ytitle=objectives[1],
        ztitle=objectives[2],
        axes_ranges=[i for j in zip(lb, ub) for i in j]
        )
    plotter.view_isometric()
    plotter.show()
    
    
    
# %%
