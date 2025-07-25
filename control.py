# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:04:52 2025

@author: hmtha
"""


import numpy as np 
from numba import njit
from datetime import datetime as dt
from collections.abc import Callable
from scipy.spatial import Delaunay
import warnings

import mh_functions as mh
from fitness import (euclideanDistance, pointwiseDistance)
from Fileprinter import Fileprinter
import termination_criteria as tc
from timekeeper import timekeeper, keeptime, PrintTimekeeper
import diversity as dy

import plotlogs as pl
on_switch=False

#%%

# =============================================================================
# TODO: log more stuff

# TODO: update fitness so as not to calculate it for optimum niche
    # Low priority - fitness is an inexpensive calculation

# =============================================================================

# =============================================================================
# Musing: very low population very high niches?? how does that behave
# =============================================================================

class Problem:
    # @keeptime("init", on_switch)
    def __init__(
            self,
            func,
            bounds, # (lb[:], ub[:])
            integrality: np.ndarray[bool] = None,
            
            x0: Callable[..., ...]|np.ndarray|str = "uniform", 

            log: str = None, # naming convention for log files
            log_freq: int = 1, # frequency of printing batched logs
            log_level: str = "detailed", 
            log_resume: bool = False, 
            constraint_violation: bool = True,
            maximize: bool = False,
            vectorized: bool = False, # whether objective function accepts vectorsied array of points
            fargs: tuple[...] = (),
            fkwargs: dict[..., ...] = {},
            
            known_optimum: np.ndarray[float] = None, # None or np.array([coords])
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
        self.vectorized = vectorized
        self.constraint_violation = constraint_violation
        self.fargs = fargs
        self.fkwargs = fkwargs
        self.x0 = x0
        self.slack = np.inf

        assert isinstance(maximize, bool)
        self.maximize = maximize
        self.noptimal_obj = -np.inf if self.maximize else np.inf

        # `bounds` is a tuple containing (lb, ub)
        self.lb, self.ub = self.bounds = bounds 
        if integrality is None:
            self.mutFunction = mh.mutGaussianFloatVec
            self.integrality = np.zeros(len(self.lb), np.bool_)
        else: 
            self.mutFunction = mh.mutGaussianMixedVec
            self.integrality = integrality if integrality is not None else np.zeros(len(self.lb), np.bool_)
            
        assert islistlike(self.lb)
        assert islistlike(self.ub)
        assert len(self.lb) == len(self.ub)
        assert len(self.integrality) == len(self.lb)
        self.bool_dtype = (((self.ub-self.lb)==1) * (self.integrality)).astype(np.bool_)
        
        self.ndim = len(self.lb)
        self.cx = mh.cxTwoPoint if self.ndim > 2 else mh.cxOnePoint

        self.centre = (self.ub - self.lb) / 2 + self.lb
        if known_optimum is None: 
            self.optimum = self.centre.copy()
        else: 
            assert islistlike(known_optimum)
            self.optimum = np.array(known_optimum)
            assert len(self.optimum) == self.ndim
        if self.vectorized:
            if self.constraint_violation:
                self.optimal_obj, self.init_violation = func(np.array([self.optimum]), *self.fargs, **self.fkwargs)[0]
            else:
                self.optimal_obj = func(np.array([self.optimum]), *self.fargs, **self.fkwargs)[0]
                self.init_violation = 0
        else:
            if self.constraint_violation:
                self.optimal_obj, self.init_violation = func(self.optimum, *self.fargs, **self.fkwargs)
            else:
                self.optimal_obj = func(self.optimum, *self.fargs, **self.fkwargs)
                self.init_violation = 0
        
        self.log = log
        self.log_freq = log_freq
        self.log_level = "detailed"
        if self.log is not None: 
            self.nobjective_printer = Fileprinter(
                file_name = self.log+"-nobjective.csv", 
                save_freq = self.log_freq, 
                header = None,
                resume = log_resume,
                create_dir = True,
                )
            self.noptimality_printer = Fileprinter(
                file_name = self.log+"-noptimality.csv", 
                save_freq = self.log_freq, 
                header = None,
                resume = log_resume,
                )
            self.nfitness_printer = Fileprinter(
                file_name = self.log+"-nfitness.csv", 
                save_freq = self.log_freq, 
                header = None,
                resume = log_resume,
                )
            self.diversity_printer = Fileprinter(
                file_name = self.log+"-diversity.csv", 
                save_freq = self.log_freq, 
                header = ["it #", "VESA", "mean of std.s", "min of std.s", 
                          "max of std.s", "mean of var.s", "min of var.s", "max of var.s", 
                          "sum of fit", "mean of fit"],
                resume = log_resume,
                )
            if self.log_level == "detailed":
                self.evolution_printer = Fileprinter(
                    file_name = self.log+"-evolution.csv", 
                    save_freq = self.log_freq, 
                    header = None, 
                    resume = log_resume)

        
    def Initiate(
            self, 
            nniche: int, 
            popsize: int,
            ):
        assert nniche >= 1
        if nniche == 1: 
            warnings.warn("One niche. Pure optimisation.", UserWarning)
        self.nniche = nniche
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
        
        self.optimal_obj += self.violation_factor * self.init_violation
        del self.init_violation
        
        # evaluate fitness of initial population
        self.Evaluate_objective()
        self.Update_optimum()
        self.Evaluate_noptimality()
        self.Evaluate_fitness()
        self.Evaluate_diversity()
        
        self._initiated = True
        
    # @keeptime("Initiate_population", on_switch)
    def Initiate_population(self, x0):
        self.population = np.empty((self.nniche, self.popsize, self.ndim))
        self.objective = np.empty((self.nniche, self.popsize), dtype=np.float64)
        self.noptimality = np.empty((self.nniche, self.popsize), dtype=np.bool_)
        self.fitness = np.empty((self.nniche, self.popsize), dtype=np.float64)
        self.centroids = np.empty((self.nniche, self.ndim), np.float64)
        self.niche_elites = np.empty((self.nniche-1, 1, self.ndim))
        self.unselfish_niche_elite = 0.0
        self.violation = np.zeros((self.nniche, self.popsize), dtype=np.float64)
        self.penalized_objective = np.empty((self.nniche, self.popsize), dtype=np.float64)

        if hasattr(x0, "__call__"):
            for i in range(self.nniche):
                for j in range(self.popsize):
                    self.population[i, j, :] = x0(self.lb, self.ub)
        if isinstance(x0, str):
            rng_dist = getattr(self.rng, x0)
            for i in range(self.nniche):
                for j in range(self.popsize):
                    self.population[i, j, :] = rng_dist(self.lb, self.ub)

        else:
            assert isinstance(x0, np.ndarray)
            assert x0.ndim == 3
            assert x0.shape[0] == self.nniche
            assert x0.shape[1] == 1 or x0.shape >= self.popsize 
            assert x0.shape[2] == self.ndim
            
            if x0.shape[1] > 1:
                njit_deepcopy(self.population, x0[:, :self.popsize, :])
            
            else: # x0.shape[1] == 1
                sigma = 0.01*(self.ub-self.lb)
                clone_parents(self.population, x0)
                self.mutFunction(self.population, sigma, 0.8, self.rng, self.integrality, self.bool_dtype)
                apply_bounds_vec(self.population, self.lb, self.ub)
                
    # @keeptime("Add_niche", on_switch)
    def Add_niche(self, new_niche):
        self.population = Add_niche_to_array(self.population, new_niche)
        self.objective = Add_niche_to_array(self.objective, new_niche)
        self.penalized_objective = Add_niche_to_array(self.penalized_objective, new_niche)
        self.fitness = Add_niche_to_array(self.fitness, new_niche)
        self.noptimality = Add_niche_to_array(self.noptimality, new_niche)
        self.violation = Add_niche_to_array(self.violation, new_niche)
        self.violation[:, :] = 0.0
        
        self.centroids = np.zeros((self.nniche + new_niche, self.ndim))
        self.niche_elites = np.empty((self.nniche + new_niche - 1, 1, self.ndim))
        self.unselfish_niche_elite = 0.0
        
        if self.nniche < self.ndim + 1 or self.ndim == 1 or self.new_niche_heuristic is False:
            for i in range(self.nniche, self.nniche + new_niche):
                for j in range(self.popsize):
                    for k in range(self.ndim):
                        self.population[i, j, k] = self.rng.uniform(self.lb[k], self.ub[k])
            
        else: 
            delaunay = Delaunay(self.centroids)
            candidates = np.empty((len(delaunay.simplices), self.ndim))
            radii = np.full(len(delaunay.simplices), np.inf)
    
            for i in range(len(delaunay.simplices)):
                simplex_pts = self.centroids[delaunay.simplices[i]]
                
                # N-dimensional circumcenter calculation
                A = 2 * (simplex_pts[1:] - simplex_pts[0])
                b = np.sum(simplex_pts[1:]**2 - simplex_pts[0]**2, axis=1)
                
                try:
                    candidates[i] = np.linalg.solve(A, b)
                    radii[i] = np.sum((simplex_pts[0] - candidates[i])**2)
                except np.linalg.LinAlgError:
                    # This occurs for degenerate simplices (e.g., collinear in 2D)
                    continue
        
            best = radii.argsort(stable=self.stable)
            best = best[~np.isinf(radii)]
            
            if new_niche < 0: 
                new_niche = min(len(best), -new_niche)
            
            for i in range(new_niche):
                nn_idx = self.nniche+new_niche-1
                self.population[nn_idx, 0] = candidates[best[-(i+1)]]
                for j in range(1, self.popsize):
                    self.population[nn_idx, j] = self.population[nn_idx, 0]

            sigma = 0.1*(self.ub-self.lb)
            self.mutFunction(self.population, sigma, 0.8, self.rng, self.integrality, self.bool_dtype)
            apply_bounds_vec(self.population, self.lb, self.ub)
        
        self.Evaluate_newniche_objectives(new_niche)
        self.nniche += new_niche
        self.Evaluate_fitness()

        if self.log is not None: 
            self.nobjective_printer._flush()
            self.noptimality_printer._flush()
            self.nfitness_printer._flush()
            self.diversity_printer._flush()
            if self.log_level == "detailed":
                self.evolution_printer._flush()

        return 
    
    def popsize_safe_select_parents(self, new_popsize):
        old_to_new = self.popsize/new_popsize
        elitek = int(self.elitek * old_to_new)
        tournk = int(self.tournk * old_to_new)
        tournsize = int(max(self.tournsize, min(2, old_to_new)))
        
        safe_parents = np.empty((self.nniche, tournk + elitek, self.ndim))
        
        _select_parents(
                safe_parents, self.population, self.fitness, self.objective, 
                self.noptimality, elitek, tournk, tournsize, 
                self.rng, self.maximize, self.stable)
        
        for i in range(self.nniche):
            self.rng.shuffle(safe_parents[i, :, :])
        clone_parents(self.parents, safe_parents)
        
        return
    
    # @keeptime("Step", on_switch)
    def Step(
            self, 
            maxiter: int|float = np.inf, # max no of iterations in this step
            popsize: int = 100, # max no of individuals in each niche
            elitek: int|float = 0.1, # number of parents selected as absolute best
            tournk: int|float = 0.9, # number of parents selected via tournament
            tournsize: int = 4, # tournament size parameter
            mutation: float|tuple[float] = (0.5, 0.75), # mutation probability
            sigma: float|tuple[float] = 0.1, # standard deviation of gaussian noise for mutation (gets scaled)
            crossover: float|tuple[float] = 0.4, # crossover probability
            slack: float = np.inf, # noptimal slack in range (1.0, inf)
            violation_factor: float = None, # penalty multiplier for constraint violation
            niche_elitism: str = None, # clone fittest parent in each niche
            new_niche: int = 0, 
            new_niche_heuristic: bool = True,
            disp_rate: int = 0,
            convergence: Callable[..., bool]|list[Callable[..., bool]] = None, 
            callback: Callable[..., ...] = None,
            ):
        if violation_factor is None: 
            self.violation_factor = float(self.constraint_violation)
        else: 
            if bool(violation_factor) != self.constraint_violation:
                warnings.warn("violation_factor ignored when constraint_violation = True", UserWarning)
            self.violation_factor = violation_factor
        
        if not self._initiated:
            if not new_niche > 0:
                raise Exception("""Supply `new_niche: int > 0` to initiate problem.""" )
            self.Initiate(
                nniche = new_niche, 
                popsize = popsize
                )
            new_niche=0
        
        # if new_niche is an int in (0, inf), then generate {new_niche}
        #     points - use delaunay for as many as possible and random generation if not
        # if new_niche is an int in (-inf, 0) then generate up to 
        #     {abs(new_niche)} points using delaunay exclusively 
        self.new_niche_heuristic = new_niche_heuristic
        if new_niche > 0:
            self.Add_niche(new_niche)

        ### Hyperparameters related to parent selection
        self.tournsize = tournsize
        assert elitek!=-1 or tournk !=-1, "only 1 of elitek and tournk may be -1"
        self.elitek = elitek if isinstance(elitek, int) else int(elitek*popsize)
        self.tournk = tournk if isinstance(tournk, int) else int(tournk*popsize)
        self.elitek = popsize - self.tournk if self.elitek == -1 else self.elitek
        self.tournk = popsize - self.elitek if self.tournk == -1 else self.tournk
        assert self.elitek + self.tournk <= popsize, "elitek + tournk should be weakly less than popsize"
        assert popsize > self.tournsize, "tournsize should be less than popsize"
        
        self.parents = np.empty((self.nniche, self.tournk + self.elitek, self.ndim), 
                                dtype=np.float64)
       
        if popsize == self.popsize or popsize is None:
            self.Select_parents()
        else: 
            self.popsize_safe_select_parents(popsize)
            # Change popsize here
            self.popsize = popsize
            self.population = np.empty((self.nniche, self.popsize, self.ndim), dtype=np.float64)
            self.objective = np.empty((self.nniche, self.popsize), dtype=np.float64)
            self.penalized_objective = np.empty((self.nniche, self.popsize), dtype=np.float64)
            self.violation = np.zeros((self.nniche, self.popsize), dtype=np.float64)
            self.noptimality = np.empty((self.nniche, self.popsize), dtype=np.bool_)
            self.fitness = np.empty((self.nniche, self.popsize), dtype=np.float64)
            
        ### Hyperparameters related to parent breeding
        self.mutation = mutation
        self.sigma = sigma # adjust sigma for unnormalised bounds
        self.crossover = crossover
        self.niche_elitism = niche_elitism
        assert niche_elitism in (None, "selfish", "unselfish")
        
        ### Near-optimal definitions
        self.slack = slack
        if self.maximize:
            self.noptimal_obj = self.optimal_obj * (1-(self.slack - 1))
        else: 
            self.noptimal_obj = self.optimal_obj*self.slack
            
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
            while not self.convergence(self): 
                self.Loop()
                if self.maximize:
                    # improve this
                    best = [round(float(max(obj)),2) for obj in self.objective]
                else:
                    best = [round(float(min(obj)),2) for obj in self.objective]
                # print("\r",
                #       f"iteration {_i}. Current_best: {best}. Time: {dt.now()-self.start_time}."
                #       , end="\r")
                if self._i % disp_rate == 0:
                    print(
                        # improve this
    f"iteration {self.nit_}. Current_best: {best}. Time: {dt.now()-self.start_time}.")

        else: 
            while not self.convergence(self): 
                self.Loop()
        self.nstep_+=1
    
    # @keeptime("Loop", on_switch)        
    def Loop(self):
        
        if self.initstep is False:
            self.Select_parents()
        self.initstep = False
        
        self.mutationInst = dither_instance(self.mutation, self.rng)
        self.crossoverInst = dither_instance(self.crossover, self.rng)
        self.sigmaInst = dither_instance(self.sigma, self.rng) * (self.ub - self.lb)
        
        self.Generate_offspring()
        self.Evaluate_objective()
        self.Update_optimum()
        self.Evaluate_noptimality()
        self.Evaluate_fitness()
        self.Evaluate_diversity()
        
# =============================================================================
        if self.log:
            self.Print_diversity()
            self.Return_noptima()
            self.nobjective_printer(np.atleast_2d(self.nobjective))
            self.noptimality_printer(np.atleast_2d(self.nnoptimality))
            self.nfitness_printer(np.atleast_2d(self.nfitness))
            if self.log_level == "detailed":
                output = np.concatenate((
                    self.nit_*np.ones((self.nniche, 1), int), 
                    np.atleast_2d(np.array(
                        ["optimum"] + [f"nopt{n}" for n in range(1, self.nniche)]
                        )).T,
                    np.atleast_2d(self.nobjective).T, 
                    np.atleast_2d(self.nfitness).T, 
                    self.noptima
                    ), axis=1)
                self.evolution_printer(output)
# =============================================================================
        if self.Callback is not None:
            self.Callback(self)
        self._i += 1
        self.nit_ += 1
            
        
    # @keeptime("Terminate", on_switch)
    def Terminate(self):
        self.Run_statistics()
        self.Return_noptima()
        if self.log is not None: 
            self.noptima_printer = Fileprinter(
                file_name = self.log+"-noptima.csv", 
                save_freq = 1, 
                header = ["optimum"] + [f"nopt{n}" for n in range(1, self.nniche)],
                resume = False,
                )
            self.noptima_printer(np.atleast_2d(self.nobjective))
            self.noptima_printer(np.atleast_2d(self.nnoptimality))
            self.noptima_printer(self.noptima.T)
        return self.noptima, self.nfitness, self.nobjective, self.nnoptimality

    # @keeptime("Update_optimum", on_switch)
    def Update_optimum(self):
    # =============================================================================
    # This can be njitted and refactored for speed up
    # =============================================================================
        if self.maximize:
            if self.constraint_violation:
                feasible_objectives = self.objective[(self.violation == 0.0)]
                if feasible_objectives.max() > self.optimal_obj:
                    for n in range(self.nniche):
                        if feasible_objectives[n].max() < self.optimal_obj:
                            continue
                        self.optimal_obj = feasible_objectives[n].max()
                        self.optimum = self.population[n, feasible_objectives[n].argmax(), :].copy()
            else:
                if self.objective.max() > self.optimal_obj:
                    for n in range(self.nniche):
                        if self.objective[n].max() < self.optimal_obj:
                            continue
                        self.optimal_obj = self.objective[n].max()
                        self.optimum = self.population[n, self.objective[n].argmax(), :].copy()
            self.noptimal_obj = self.optimal_obj * (1-(self.slack - 1))
        else: 
            if self.constraint_violation:
                feasible_objectives = self.objective[(self.violation == 0.0)]
                if feasible_objectives.min() > self.optimal_obj:
                    for n in range(self.nniche):
                        if feasible_objectives[n].min() < self.optimal_obj:
                            continue
                        self.optimal_obj = feasible_objectives[n].min()
                        self.optimum = self.population[n, feasible_objectives[n].argmin(), :].copy()
            else:
                if self.objective.min() < self.optimal_obj:
                    for n in range(self.nniche):
                        if self.objective[n].min() > self.optimal_obj:
                            continue
                        self.optimal_obj = self.objective[n].min()
                        self.optimum = self.population[n, self.objective[n].argmin(), :].copy()
            self.noptimal_obj = self.optimal_obj*self.slack


    # @keeptime("Return_noptima", on_switch)
    def Return_noptima(self):
        if self.maximize:
            nindex = [self.objective[0][self.violation[0] == 0].argmax()]
        else: 
            nindex = [self.objective[0][self.violation[0] == 0].argmin()]
            
        nindex += list(best_nindex(self.fitness, self.noptimality, self.violation))
            
        self.noptima = np.array(
            [self.population[0, nindex[0]]] + [self.population[n, nindex[n]] for n in range(1, self.nniche)]
            )

        self.nfitness = np.array([self.fitness[n, nindex[n]] for n in range(self.nniche)])
        self.nobjective = np.array([self.objective[n, nindex[n]] for n in range(self.nniche)])
        self.nnoptimality = np.array([self.noptimality[n, nindex[n]] for n in range(self.nniche)])
        
    
    def Run_statistics(self):
        """TODO"""
        pass

    def Select_parents(self):
        _select_parents(
                self.parents, self.population, self.fitness, self.penalized_objective, 
                self.noptimality, self.elitek, self.tournk, self.tournsize, 
                self.rng, self.maximize, self.stable)
    
    # @keeptime("Generate_offspring", on_switch)
    def Generate_offspring(self):
        if self.niche_elitism == "selfish":
            self.niche_elites = populate_niche_elites(
                self.niche_elites, self.population, self.fitness, self.noptimality, self.penalized_objective, self.maximize)
        elif self.niche_elitism == "unselfish":
            niche_elites = np.empty_like(self.niche_elites)
            fitness = np.empty((self.nniche-1, 1))
            niche_elites = populate_niche_elites(
                self.niche_elites, self.population, self.fitness, self.noptimality, self.penalized_objective, self.maximize)
            _evaluate_fitness(fitness, niche_elites, np.vstack((self.optimum, niche_elites[:, 0, :])))
            if np.mean(fitness) > self.unselfish_niche_elite:
                self.unselfish_niche_elite = np.mean(fitness)
                njit_deepcopy(self.niche_elites, niche_elites)
        clone_parents(self.population, self.parents)
        
        mh.cx_vec(self.population, self.crossoverInst, self.cx, self.rng)
        self.mutFunction(self.population, self.sigmaInst, self.mutationInst, self.rng, self.integrality, self.bool_dtype)
        apply_bounds_vec(self.population, self.lb, self.ub)
        
        self.population[0, 0, :] = self.optimum
        
        if self.niche_elitism in ("selfish", "unselfish"):
            for i in range (1, self.nniche):
                self.population[i, 0, :] = self.niche_elites[i-1, :]

    def Evaluate_fitness(self):
        """
        Calculates the fitness of each individual as the minimum (per-variable)
        distance from the centroids of all the other niches besides the one 
        where the individual itself sits, and including the 'optimal' niche. 
        """
        find_centroids(self.centroids, self.population)
        _evaluate_fitness(
            self.fitness, self.population, self.centroids)

    # @keeptime("Evaluate_objective", on_switch)        
    def Evaluate_objective(self):
        """
        Calculates the objective function for each individual
        """
        if self.vectorized:
            if self.constraint_violation:
                for n in range(self.nniche):
                    self.objective[n], self.violation[n] = self.func(
                        self.population[n], *self.fargs, **self.fkwargs)
            else:
                for n in range(self.nniche):
                    self.objective[n] = self.func(self.population[n], *self.fargs, **self.fkwargs)
        else: 
            if self.constraint_violation:
                for n in range(self.nniche):
                    for i in range(self.popsize):
                        self.objective[n, i], self.violation[n, i] = self.func(
                            self.population[n, i], *self.fargs, **self.fkwargs)
            else:
                for n in range(self.nniche):
                    for i in range(self.popsize):
                        self.objective[n, i] = self.func(self.population[n, i], *self.fargs, **self.fkwargs)
        Apply_penalties(self.penalized_objective, self.objective, self.violation, self.violation_factor)
                
    def Evaluate_newniche_objectives(self, new_niche):
        if self.vectorized:
            if self.constraint_violation: 
                for n in range(self.nniche, self.nniche + new_niche):
                    self.objective[n], self.violation[n] = self.func(
                        self.population[n], *self.fargs, **self.fkwargs)
            else:
                for n in range(self.nniche, self.nniche + new_niche):
                    self.objective[n] = self.func(self.population[n], *self.fargs, **self.fkwargs)
        else: 
            if self.constraint_violation: 
                for n in range(self.nniche, self.nniche + new_niche):
                    for i in range(self.popsize):
                        self.objective[n, i], self.violation[n, i] = self.func(
                            self.population[n, i], *self.fargs, **self.fkwargs)
                    self.violation *= self.violation_factor
                    self.penalized_objective = self.objective + self.violation
            else:
                for n in range(self.nniche, self.nniche + new_niche):
                    for i in range(self.popsize):
                        self.objective[n, i] = self.func(self.population[n, i], *self.fargs, **self.fkwargs)
        Apply_penalties(self.penalized_objective, self.objective, self.violation, self.violation_factor)

    def Evaluate_noptimality(self):
        _evaluate_noptimality(
            self.noptimality, self.population, self.objective, self.noptimal_obj, self.maximize)
   
    def Print_diversity(self):
        result = [self.nit_]
        result.append(dy.VESA(self.centroids))
        _std = dy.std(self.fitness, self.noptimality)
        result.append(np.mean(_std))
        result.append(np.min(_std))
        result.append(np.max(_std))
        _var = dy.var(self.fitness, self.noptimality)
        result.append(np.mean(_var))
        result.append(np.min(_var))
        result.append(np.max(_var))
        result.append(dy.sumOfFitness(self.fitness, self.noptimality))
        result.append(dy.meanOfFitness(self.fitness, self.noptimality))

        self.diversity_printer(np.atleast_2d(np.array(result)))
        
    def Evaluate_diversity(self):
        self._vesa = dy.VESA(self.centroids)
        # Should this be noptimality * (violation>0)??
        _std = dy.std(self.fitness, self.noptimality)
        self._mean_std_fit = np.mean(_std)
        _var = dy.var(self.fitness, self.noptimality)
        self._mean_var_fit = np.mean(_var)
        self._mean_fit = dy.meanOfFitness(self.fitness, self.noptimality)
    
#%%
    
def islistlike(obj):
    if not hasattr(obj, "__iter__"):
        return False
    if not hasattr(obj, "__len__"):
        return False
    return True

def isfunclike(obj):
    # Add other conditions?
    return callable(obj)

@njit
def Apply_penalties(penalized_objective, objective, violation, violation_factor):
    penalized_objective[:, :] = objective + violation * violation_factor # inplace - propogates up to Problem

@njit
def Add_niche_to_array(old_array, new_niche):
    ndim = old_array.ndim
    if ndim == 3:
        new_array = np.empty(
            (old_array.shape[0] + new_niche, 
             old_array.shape[1], 
             old_array.shape[2],
             ), 
            dtype=old_array.dtype)
        for i in range(old_array.shape[0]):
            for j in range(old_array.shape[1]):
                for k in range(old_array.shape[2]):
                    new_array[i, j, k] = old_array[i, j, k]
    elif ndim == 2:
        new_array = np.empty(
            (old_array.shape[0] + new_niche, 
             old_array.shape[1], 
             ), 
            dtype=old_array.dtype)
        for i in range(old_array.shape[0]):
            for j in range(old_array.shape[1]):
                    new_array[i, j] = old_array[i, j]
    else: 
        raise Exception
    return new_array

@njit 
def best_nindex(fitness, noptimality, violation):
    # first niche reserved for optimisation
    feasible_mask = (violation == 0) * noptimality
    indices = np.zeros(fitness.shape[0] - 1, np.int64)
    for i in range(len(indices)):
        i_o = i + 1 
        _best = -np.inf
        if feasible_mask[i_o].any():
            for j in range(fitness.shape[1]):
                if feasible_mask[i_o, j] is True: 
                    if fitness[i_o, j] > _best:
                        _best = fitness[i_o, j]
                        indices[i] = j
        else: 
            for j in range(fitness.shape[1]):
                if fitness[i_o, j] > _best:
                    _best = fitness[i_o, j]
                    indices[i] = j
    return indices

@njit 
def njit_deepcopy(new, old):
    flat_new = new.ravel()
    flat_old = old.ravel()
    
    for i in range(flat_old.shape[0]):
        flat_new[i] = flat_old[i]
    
# @keeptime("Evaluate_fitness", on_switch)        
@njit
def _evaluate_fitness(fitness, population, centroids):
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            fitness[i, j] = np.inf
            for c in range(population.shape[0]):
                if i == c: 
                    continue
                fitness[i, j] = min(
                    fitness[i, j], 
                    euclideanDistance(population[i, j], centroids[c])
                    )

# @keeptime("Select_parents", on_switch)
@njit
def _select_parents(
        parents, population, fitness, objective, noptimality, elitek, tournk, tournsize, rng, maximize, stable):
    
    parents[0, :elitek, :] = mh.selBest(population[0, :, :], objective[0, :], elitek, maximize, stable)
    parents[0, elitek:, :] = mh.selTournament(
        population[0, :, :], objective[0, :], tournk, tournsize, rng, maximize)
        
    for i in range(1, parents.shape[0]):
        parents[i, :elitek, :] = mh.selBest_fallback(
            population[i, :, :], fitness[i, :], noptimality[i, :], objective[i, :], elitek, maximize, stable)
        parents[i, elitek:, :] = mh.selTournament_fallback(
            population[i, :, :], fitness[i, :], noptimality[i, :], objective[i, :], tournk, tournsize, rng, maximize)

# @keeptime("Evaluate_noptimality", on_switch)        
@njit
def _evaluate_noptimality(noptimality, population, objective, noptimal_obj, maximize):
    if maximize:
        for i in range(population.shape[0]):
            for j in range(population.shape[1]):
                noptimality[i, j] = objective[i, j] > noptimal_obj
    else: 
        for i in range(population.shape[0]):
            for j in range(population.shape[1]):
                noptimality[i, j] = objective[i, j] < noptimal_obj
    
@njit
def njit_cdist(A, B):
    """
    Numba-compatible implementation of Euclidean distance calculation,
    similar to the output of scipy.cdist(A, B, 'euclidean').
    Args:
        A (np.array): An (M, K) array of M vectors.
        B (np.array): An (N, K) array of N vectors.
    Returns:
        np.array: An (M, N) matrix of Euclidean distances.
    """
    M, K = A.shape
    N, K = B.shape
    # Calculate the squared L2 norm for each row in A and B
    A_sq = np.sum(A**2, axis=1).reshape(M, 1)
    B_sq = np.sum(B**2, axis=1).reshape(1, N)
    # Calculate the dot product term: 2 * (A @ B.T)
    dot_product = 2 * A @ B.T
    # Calculate squared Euclidean distances using the identity
    dist_sq = A_sq - dot_product + B_sq
    # Handle potential floating-point inaccuracies that result in small negative values
    dist_sq[dist_sq < 0] = 0
    return np.sqrt(dist_sq)

@njit
def populate_niche_elites(niche_elites, population, fitness, noptimality, p_objective, maximize):
    for i in range(1, 1 + niche_elites.shape[0]):
        niche_elites[i-1, 0, :] = mh.selBest1_fallback(
            population[i], fitness[i], noptimality[i], p_objective[i], maximize)
    return niche_elites

# @keeptime("find_centroids", on_switch)
@njit
def find_centroids(centroids, population):
    centroids[:,:] = 0.0
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                centroids[i, k] += population[i, j, k]
    centroids /= population.shape[1]

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
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                population[i, j, k] = min(ub[k], max(lb[k], population[i, j, k]))

@njit
def apply_integrality(population, integrality, lb, ub):
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                if not integrality[k]:
                    continue
                population[i, j, k] = round(population[i, j, k])

@njit
def clone_parents(population, parents):
    nparents = parents.shape[1]
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            jn = j%nparents
            for k in range(population.shape[2]):
                population[i, j, k] = parents[i, jn, k]
    


#%% 


    
if __name__ == "__main__":
    
    @njit
    def Objective(values_array): 
        """ Vectorized = True """
        z = 2 + np.zeros(values_array.shape[0], np.float64)
        for i in range(values_array.shape[0]):
            for j in range(values_array.shape[1]):
                z[i] += np.sin(19 * np.pi * values_array[i, j]) + values_array[i, j] / 1.7
        return z
        
    @njit
    def feasibility_wrapper(values_array):
        objective = Objective(values_array)
        constraint_violation = np.zeros(objective.shape, np.float64)
        return objective, constraint_violation
    
    # def Objective(x):
    #     return ((0.5*(x-0.55)**2 + 0.9)*np.sin(5*np.pi*x)**6)[0]
    
    lb = 0*np.ones(2) 
    ub = 1*np.ones(2)
    
    FILEPREFIX = "logs/testprob-z"
    
    problem = Problem(
        # func=feasibility_wrapper,
        func=Objective,
        bounds = (lb, ub),
        constraint_violation = False,
        # constraint_violation = True,
        maximize = True, 
        vectorized = True,
        log = FILEPREFIX,
        log_freq = 500,
        # random_seed = 1,
        # x0 = x0,
        )

    NNICHE = 6

    problem.Step(
        maxiter=np.inf, 
        popsize=int(500/NNICHE), 
        elitek=0.055,
        tournk=-1,
        tournsize=2,
        mutation=0.1, 
        sigma=1.0,
        crossover=0.4, 
        slack=1.12,
        disp_rate=25,
        niche_elitism = "unselfish", 
        new_niche = NNICHE,
        convergence = tc.MultiConvergence( 
            [tc.FixedValue(
                0.0001, 
                maximize=False, 
                attribute="_mean_var_fit"
                ),
             tc.GradientStagnation(
                 window=50, 
                 improvement=0.01/50,
                 maximize=True, 
                 attribute="_vesa",
                 ),
            ],
            how="and",
            )
        )
    problem.Step(
        maxiter=np.inf, 
        popsize=int(500/NNICHE), 
        elitek=0.055,
        tournk=-1,
        tournsize=2,
        mutation=0.1, 
        sigma=1.0,
        crossover=0.4, 
        slack=1.12,
        disp_rate=25,
        niche_elitism = "unselfish", 
        new_niche = NNICHE,
        new_niche_heuristic = False,
        convergence = tc.MultiConvergence( 
            [tc.FixedValue(
                0.0001, 
                maximize=False, 
                attribute="_mean_var_fit"
                ),
             tc.GradientStagnation(
                 window=50, 
                 improvement=0.01/50,
                 maximize=True, 
                 attribute="_vesa",
                 ),
            ],
            how="and",
            )
        )
    
    
    noptima, nfitness, nobjective, nnoptimality = problem.Terminate()
    for n in range(problem.nniche): 
        print(noptima[n], nfitness[n], nobjective[n], nnoptimality[n])


        
    # problem.Step(
    #     maxiter=10, 
    #     popsize=200, 
    #     elitek=0.25,
    #     tournk=-1,
    #     tournsize=2,
    #     mutation=0.5, 
    #     sigma=0.4,
    #     crossover=0.3, 
    #     slack=1.12,
    #     disp_rate=25,
    #     niche_elitism=True, 
    #     new_niche = 3,
    #     )
    # noptima, nfitness, nobjective, nnoptimality = problem.Terminate()
    # for n in range(problem.nniche): 
    #     print(noptima[n], nfitness[n], nobjective[n], nnoptimality[n])
        
        
    # # PrintTimekeeper(on_switch=on_switch)
    # print("-"*20)
    # problem.Step(
    #     maxiter=200, 
    #     popsize=300, 
    #     elitek=0.2,
    #     tournk=-1,
    #     tournsize=2,
    #     mutation=0.4, 
    #     sigma=0.15,
    #     crossover=0.3, 
    #     slack=1.12,
    #     # new_niche=1,
    #     disp_rate=20,
    #     niche_elitism=True, 
    #     )
    # noptima, nfitness, nobjective, nnoptimality = problem.Terminate()
    # for n in range(problem.nniche): 
    #     print(noptima[n], nfitness[n], nobjective[n], nnoptimality[n])
    # # PrintTimekeeper(on_switch=on_switch)
    # print("-"*20)

    # problem.Step(
    #     maxiter=10, 
    #     popsize=1000, 
    #     elitek=0.4,
    #     tournk=-1,
    #     tournsize=2,
    #     mutation=0.5, 
    #     sigma=0.005,
    #     crossover=0.3, 
    #     slack=1.12,
    #     # new_niche=1,
    #     disp_rate=5,
    #     niche_elitism=True, 
    #     )
    # noptima, nfitness, nobjective, nnoptimality = problem.Terminate()
    # for n in range(problem.nniche): 
    #     print(noptima[n], nfitness[n], nobjective[n], nnoptimality[n])
    # # PrintTimekeeper(on_switch=on_switch)

        
    PrintTimekeeper(on_switch=on_switch)
    
    # pl.plot_opt_path_2d(FILEPREFIX)
    pl.plot_noptima(FILEPREFIX)
    pl.plot_stat_evolution(FILEPREFIX)
    pl.plot_vesa(FILEPREFIX)
    

