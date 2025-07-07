# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:43:32 2025

@author: u6942852
"""
import numpy as np
from numba import njit
from collections.abc import Callable, Sequence
from functools import partial 
from termination_criteria import FixedValue
from datetime import datetime as dt
from datetime import timedelta as td
from tqdm import tqdm

from control import Problem

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

class Tuning:
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
        score = self.func(self.values)
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


#%%
if __name__=="__main__":

    @njit
    def Objective(values_array): 
        """ Vectorized = True """
        # For the first test, ndim=2 and works for a function with two decision variables
        z = 2 + np.zeros(values_array.shape[0], np.float64)
        for i in range(values_array.shape[0]):
            for j in range(values_array.shape[1]):
                z[i] += np.sin(19 * np.pi * values_array[i, j]) + values_array[i, j] / 1.7
        return z   

    def Optimize(hyperparameters):
        
        problem = Problem(
            Objective,
            bounds = (lb, ub),
            x0 = "uniform", 
            maximize = True,
            vectorized = True,
            fargs = (),
            fkwargs = {},
            random_seed = 1,
            )

        start = dt.now()

        problem.Step(
            **hyperparameters, 
            new_niche=4,
            slack=1.12, 
            disp_rate=0,
            convergence = FixedValue(5.145, maximize=True, attribute="optimal_obj"), 
            )
        
        return dt.now() - start

    lb = 0*np.ones(2)
    ub = 1*np.ones(2)

    tuning = Tuning(Optimize, False, 5)

    tuning.AddParameter("popsize", int, 50, 1000)
    tuning.AddParameter("niche_elitism", bool)
    tuning.AddParameter("elitek", float, 0, 1)
    tuning.AddParameter("tournk", float, 0, 1)
    tuning.AddParameter("tournsize", int, 2, 10)
    tuning.AddParameter("mutation", float, 0, 0.9, "optional")
    tuning.AddParameter("sigma", float, 0, 0.5, "optional")
    tuning.AddParameter("crossover", float, 0, 0.9, "optional")

    tuning.AddInterdependence(["tournk", "elitek"], 1, sum, "all", "clip", "le")
    
    # Compile jit
    tuning.DrawParameters()
    tuning.RunObj()
        
    for i in tqdm(range(1000)):
        tuning.DrawParameters()
        tuning.RunObj()
    
    print(tuning.best_score)
    print(tuning.best_set[0])
