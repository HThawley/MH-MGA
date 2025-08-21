# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:25:05 2025

@author: u6942852
"""

import numpy as np 
from numba import njit
from datetime import datetime as dt
from collections.abc import Callable, Iterable
import warnings

import mh_functions as mh
import termination_criteria as tc
from timekeeper import timekeeper, keeptime, PrintTimekeeper

import plotlogs as pl
on_switch=False

#%%



class MOProblem:
    # @keeptime("init", on_switch)
    def __init__(
            self,
            func,
            bounds, # (lb[:], ub[:])
            n_objs,
            
            integrality: Iterable[bool] = None,
            
            x0: Callable[..., ...]|np.ndarray|str = "uniform", 

            feasibility: bool = False,
            maximize: bool|Iterable[bool] = False,
            vectorized: bool = False, # whether objective function accepts vectorsied array of points
            fargs: tuple[...] = (),
            fkwargs: dict[..., ...] = {},
            random_seed = None,
            # **kwargs,
            ):
        self._initiated = False
        
        self.rng = np.random.default_rng(random_seed)
        self.stable = random_seed != None
        if random_seed is not None and callable(x0):
            warnings.warn("""random_seed was supplied but a generator was supplied to `x0` 
                          which may be random but is not seeded. Try passing the name of 
                          a `numpy.random.Generator` distribution as a string""", UserWarning)
        
        self.func = func
        self.n_objs = n_objs
        self.vectorized = vectorized
        self.fargs = fargs
        self.fkwargs = fkwargs
        self.x0 = x0

        self.lb, self.ub = self.bounds = bounds 
        assert islistlike(self.lb)
        assert islistlike(self.ub)
        assert len(self.lb) == len(self.ub)
        
        self.ndim = len(self.lb)

        if isinstance(maximize, bool):
            self.maximize = np.array([maximize] * self.n_objs)
        else:
            assert islistlike(maximize)
            for m in maximize: 
                assert isinstance(m, (bool, np.bool_))
            assert len(maximize) == self.n_objs
            assert np.array(maximize).ndim == 1
            self.maximize = np.array(maximize)
        
        assert isinstance(feasibility, bool)
        self.feasibility = feasibility
            
        if integrality is None:
            self.mutFunction = mutGaussianFloatVec
            self.integrality = np.zeros(len(self.lb), np.bool_)
        else: 
            self.mutFunction = mutGaussianMixedVec
            if isinstance(integrality, bool):
                self.integrality = np.array([integrality] * self.ndim)
            else:
                assert islistlike(integrality)
                for i in integrality: 
                    assert isinstance(i, (bool, np.bool_))
                assert len(integrality) == self.ndim
                assert np.array(integrality).ndim == 1
                self.integrality = np.array(integrality)
            
        self.bool_dtype = (((self.ub-self.lb)==1) * (self.integrality)).astype(np.bool_)

        self.cx = mh.cxTwoPoint if self.ndim > 2 else mh.cxOnePoint
        self._n_pareto_current = 0


#%%
    # @keeptime("Step", on_switch)
    def Step(
            self, 
            npareto: int = None,
            maxiter: int|float = np.inf, # max no of iterations in this step
            popsize: int = 100, # max no of individuals in each niche
            mutation: float|tuple[float] = (0.5, 0.75), # mutation probability
            sigma: float|tuple[float] = 0.1, # standard deviation of gaussian noise for mutation (gets scaled)
            crossover: float|tuple[float] = 0.4, # crossover probability
            disp_rate: int = 0,
            convergence: Callable[..., bool]|list[Callable[..., bool]] = None, 
            callback: Callable[..., ...] = None,
            ):
        
        self.npareto = npareto if npareto is not None else popsize
            
        if not self._initiated:
            self.Initiate(popsize = popsize)
        
        if popsize == self.popsize or popsize is None:
            self.Select_parents()
        else: 
            assert popsize > self.popsize, "Reducing popsize may lose pareto frontier points"
            # Change popsize here
            self.popsize = popsize
            self.population = np.empty((self.popsize, self.ndim), dtype=np.float64)
            self.objective = np.empty((self.popsize, self.n_objs), dtype=np.float64)
            self.penalized_objective = np.empty((self.popsize, self.n_objs), dtype=np.float64)
            self.feasible = np.zeros((self.popsize, self.n_objs), dtype=np.bool_)
            
            self.Select_parents()

        ### Hyperparameters related to parent breeding
        self.mutation = mutation
        self.sigma = sigma # adjust sigma for unnormalised bounds
        self.crossover = crossover
        
        ### Program Control
        if convergence is None:
            self.convergence = tc.Maxiter(maxiter)
        elif hasattr(convergence, "__iter__"):
            self.convergence = tc.MultiConvergence(
                [tc.Maxiter(maxiter), 
                 *convergence,
                 ]
                )
        else: 
            self.convergence = tc.MultiConvergence(
                [tc.Maxiter(maxiter), 
                 convergence,
                 ]
                )
        self.Callback = callback
        
        self.initstep = True
        self._i = 0
        
        if disp_rate > 0:
            while (
                    (not self.convergence(self))
                    and (self.pareto.shape[0] < self.npareto)
                    ): 
                self.Loop()
                # if self.maximize:
                #     # improve this
                #     best = [round(float(max(obj)),2) for obj in self.objective]
                # else:
                #     best = [round(float(min(obj)),2) for obj in self.objective]
                # print("\r",
                #       f"iteration {_i}. Current_best: {best}. Time: {dt.now()-self.start_time}."
                #       , end="\r")
                if self._i % disp_rate == 0:
                    print(
                        # improve this
    f"iteration {self.nit_}. Pareto front: {self.pareto.shape[0]}/{self.npareto}. Time: {dt.now()-self.start_time}.")

        else: 
            while  (
                    (not self.convergence(self))
                    and (self.pareto.shape[0] <= self.npareto)
                    ):  
                self.Loop()
        self.nstep_+=1

    def Initiate(
            self, 
            popsize: int,
            ):
        
        self.popsize = popsize

        self.start_time = dt.now()
        self.nstep_ = 0
        self.nit_ = 0

        ## Initiate population
        # self.population is list of lists like a 3d array of shape (nniche, popsize, ndim)        
        self.Initiate_population(self.x0)
        del self.x0
        apply_integrality(self.population, self.integrality, self.lb, self.ub)
        apply_bounds_vec(self.population, self.lb, self.ub)
        
        # evaluate fitness of initial population
        self.Evaluate_objective()
        
        self._initiated = True
        
    # @keeptime("Initiate_population", on_switch)
    def Initiate_population(self, x0):
        self.population = np.empty((self.popsize, self.ndim))
        self.objective = np.empty((self.popsize, self.n_objs), dtype=np.float64)
        self.feasible = np.zeros((self.popsize, self.n_objs), dtype=np.bool_)

        if hasattr(x0, "__call__"):
            for j in range(self.popsize):
                self.population[j, :] = x0(self.lb, self.ub)
        if isinstance(x0, str):
            rng_dist = getattr(self.rng, x0)
            for j in range(self.popsize):
                self.population[j, :] = rng_dist(self.lb, self.ub)

        else:
            assert isinstance(x0, np.ndarray)
            assert x0.ndim == 2
            assert x0.shape[0] == 1 or x0.shape >= self.popsize 
            assert x0.shape[1] == self.ndim
            
            if x0.shape[0] > 1:
                njit_deepcopy(self.population, x0[:, :self.popsize, :])
            
            else: # x0.shape[1] == 1
                sigma = 0.01*(self.ub-self.lb)
                clone_parents(self.population, x0)
                self.mutFunction(self.population, sigma, 0.8, self.rng, self.integrality, self.bool_dtype)
                apply_bounds_vec(self.population, self.lb, self.ub)
                
    # @keeptime("Evaluate_objective", on_switch)        
    def Evaluate_objective(self):
        """
        Calculates the objective function for each individual
        """
        if self.vectorized:
            if self.feasibility:
                self.objective[self._n_pareto_current:], self.feasible[self._n_pareto_current:] = self.func(
                    self.population[self._n_pareto_current:], *self.fargs, **self.fkwargs)
            else:
                self.objective[self._n_pareto_current:] = self.func(self.population[self._n_pareto_current:], *self.fargs, **self.fkwargs)
        else: 
            if self.feasibility:
                for j in range(self._n_pareto_current, self.popsize):
                    self.objective[j, :], self.feasible[j, :] = self.func(
                        self.population[j, :], *self.fargs, **self.fkwargs)
            else:
                for j in range(self._n_pareto_current, self.popsize):
                    self.objective[j, :] = self.func(self.population[j, :], *self.fargs, **self.fkwargs)
                    
               
    def Select_parents(self):
        self.pareto, self.pareto_objs = _select_pareto(self.population, self.objective, self.maximize, self.feasible)
        self._n_pareto_current = self.pareto.shape[0]
    
    # @keeptime("Loop", on_switch)        
    def Loop(self):
        
        if self.initstep is False:
            self.Select_parents()
            if self.pareto.shape[0] == self.npareto:
                return 
        self.initstep = False
        
        self.mutationInst = dither_instance(self.mutation, self.rng)
        self.crossoverInst = dither_instance(self.crossover, self.rng)
        self.sigmaInst = dither_instance(self.sigma, self.rng) * (self.ub - self.lb)
        
        self.Generate_offspring()
        self.Evaluate_objective()
        
        if self.Callback is not None:
            self.Callback(self)
        self._i += 1
        self.nit_ += 1
        
    # @keeptime("Terminate", on_switch)
    def Terminate(self):
        self.Run_statistics()
        self.pareto, self.pareto_objs = _select_pareto(self.population, self.objective, self.maximize, self.feasible)
        return self.pareto, self.pareto_objs

    def Run_statistics(self):
        """TODO"""
        pass

    # @keeptime("Generate_offspring", on_switch)
    def Generate_offspring(self):
        clone_parents(self.population, self.pareto)
        
        cx_vec(self.population, self.crossoverInst, self.cx, self.rng, self._n_pareto_current)
        self.mutFunction(self.population, self.sigmaInst, self.mutationInst, self.rng, self.integrality, self.bool_dtype, self._n_pareto_current)
        apply_bounds_vec(self.population, self.lb, self.ub)
        
        
#%%

def islistlike(obj):
    if not hasattr(obj, "__iter__"):
        return False
    if not hasattr(obj, "__len__"):
        return False
    return True

@njit 
def njit_deepcopy(new, old):
    flat_new = new.ravel()
    flat_old = old.ravel()
    
    for i in range(flat_old.shape[0]):
        flat_new[i] = flat_old[i]
    
# @keeptime("Select_parents", on_switch)
@njit
def _select_pareto(population, objective, maximize, feasible):
    """
    Select pareto efficeint solutions as parents
    """
    popsize, n_objs = objective.shape
    # Correctly handle minimization/maximization

    feasible_mask = feasible.all(axis=1)
    feasible_indices = np.where(feasible_mask)[0]

    if feasible_indices.size == 0:
         return (np.empty((0, population.shape[1]), population.dtype), 
                np.empty((0, n_objs), objective.dtype))
    
    nfeas = feasible_indices.shape[0]

    processed_obj = np.empty((nfeas, n_objs), dtype=objective.dtype)
    
    for j in range(nfeas):
        for n in range(n_objs):
            if maximize[n]:
                processed_obj[j, n] = -objective[feasible_indices[j], n]
            else:
                processed_obj[j, n] = objective[feasible_indices[j], n]

    pareto_indices_local = np.zeros(nfeas, dtype=np.int64)
    pareto_count = 0

    # Main Loop: Iterate through each candidate solution
    for j in range(nfeas):
        candidate_obj = processed_obj[j]
        is_candidate_dominated = False
        # Track which of the *current* Pareto solutions are dominated by the new candidate. 
        dominated_in_current_front_mask = np.zeros(pareto_count, dtype=np.bool_)
        #  Inner Loop: Compare candidate against the current Pareto front 
        for k in range(pareto_count):
            current_pareto_local_idx = pareto_indices_local[k]
            current_pareto_obj = processed_obj[current_pareto_local_idx]
            
            candidate_is_better = False
            pareto_is_better = False
            for n in range(n_objs):
                if candidate_obj[n] < current_pareto_obj[n]:
                    candidate_is_better = True
                elif current_pareto_obj[n] < candidate_obj[n]:
                    pareto_is_better = True
            
            if not candidate_is_better and not pareto_is_better: # They are equal
                is_candidate_dominated = True
                break
            elif pareto_is_better and not candidate_is_better: # Pareto point dominates candidate
                is_candidate_dominated = True
                break
            elif candidate_is_better and not pareto_is_better: # Candidate dominates Pareto point
                dominated_in_current_front_mask[j] = True
            
                # Update Pareto Front
        if is_candidate_dominated:
            continue
        # The candidate is not dominated. We now rebuild the Pareto set by:
        # 1. Keeping the old members that were NOT dominated by the candidate.
        # 2. Adding the new candidate.
        # This copies over the surviving indices to the front of the array.
        write_idx = 0
        for k in range(pareto_count):
            if not dominated_in_current_front_mask[k]:
                pareto_indices_local[write_idx] = pareto_indices_local[k]
                write_idx += 1
        # Add the new candidate's index to the end of the filtered list.
        pareto_indices_local[write_idx] = j
        # Update the count of solutions on the front.
        pareto_count = write_idx + 1
    final_local_indices = pareto_indices_local[:pareto_count]
    
    pareto = np.empty((pareto_count, population.shape[1]), population.dtype)
    pareto_objs = np.empty((pareto_count, n_objs), objective.dtype)

    final_indices = feasible_indices[final_local_indices]
    pareto[:] = population[final_indices, :]
    pareto_objs[:] = objective[final_indices, :]
    
    return pareto, pareto_objs

def dither_instance(parameter, rng):
    if hasattr(parameter, "__iter__"):
        return _dither(*parameter, rng)
    else:
        return parameter

@njit
def _dither(lower, upper, rng):
    return rng.uniform(lower, upper)

# @keeptime("apply_bounds", on_switch)
@njit
def apply_bounds(ind, lb, ub):
    for i in range(len(ind)):
        ind[i] = max(lb[i], ind[i])
        ind[i] = min(ub[i], ind[i])
    return ind

@njit
def apply_bounds_vec(population, lb, ub):
    for j in range(population.shape[0]):
        for k in range(population.shape[1]):
            population[j, k] = min(ub[k], max(lb[k], population[j, k]))

@njit
def apply_integrality(population, integrality, lb, ub):
    for j in range(population.shape[0]):
        for k in range(population.shape[1]):
            if not integrality[k]:
                continue
            population[j, k] = round(population[j, k])

@njit
def clone_parents(population, parents):
    nparents = parents.shape[0]
    for j in range(population.shape[0]):
        jn = j%nparents
        for k in range(population.shape[1]):
            population[j, k] = parents[jn, k]

@njit
def cx_vec(population, crossoverInst, cx, rng, startidx=0):
    # shuffle parents for mating
    rng.shuffle(population[startidx:])
    
    for ind1, ind2 in zip(population[startidx:][::2], population[startidx:][1::2]):
        # crossover probability
        if rng.random() < crossoverInst:
            # Apply crossover
            cx(ind1, ind2, rng)

@njit 
def mutGaussianMixedVec(population, sigma, indpb, rng, integrality, boolean_mask, startidx=0):
    for j in range(startidx, population.shape[0]):
        for k in range(population.shape[1]):
            if rng.random() < indpb:
                if boolean_mask[k]:
                    population[j, k] = mh.mutBool(population[j, k]) 
                elif integrality[k]:
                    population[j, k] = mh.mutInt(population[j, k], sigma[k], rng)
                else: 
                    population[j, k] = mh.mutFloat(population[j, k], sigma[k], rng)

@njit
def mutGaussianFloatVec(population, sigma, indpb, rng, integrality, boolean_mask, startidx=0):
    for j in range(startidx, population.shape[0]):
        for k in range(population.shape[1]):
            if rng.random() < indpb:
                population[j, k] = mh.mutFloat(population[j, k], sigma[k], rng)

#%% 


    
if __name__ == "__main__":
    
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
    
    # def Objective(x):
    #     return ((0.5*(x-0.55)**2 + 0.9)*np.sin(5*np.pi*x)**6)[0]
    
    lb = 0*np.ones(2) 
    ub = 1*np.ones(2)
    
    FILEPREFIX = "logs/testprob-z"

    # THings that could be done
    ## 
    ## Min/max all objectives + min/max random weighted sums (HSJ sort of way)
    ##      Find a pareto frontier
    

    problem = MOProblem(
        # func=MOObjective,
        func=feasibility_wrapper,
        bounds = (lb, ub), 
        n_objs = 3,
        
        feasibility = True,
        maximize = True, 
        vectorized = True,
    )

    params = dict(
        npareto = 1000,
        maxiter = 100,
        popsize = 1000,
        mutation = 0.3,
        sigma = 0.5,
        crossover = 0.3,
        disp_rate=1,
        convergence=None,
        callback=None,
        )
    
    problem.Step(**params)

    pareto_points, pareto_objectives = problem.Terminate()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1])
    plt.figure()
    plt.scatter(pareto_objectives[:, 1], pareto_objectives[:, 2])
    plt.figure()
    plt.scatter(pareto_objectives[:, 2], pareto_objectives[:, 0])
    
    
