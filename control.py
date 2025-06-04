# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:04:52 2025

@author: hmtha
"""

import numpy as np 
import pandas as pd 

import utils 

from termination_criteria import Maxiter, Convergence
from post_process import return_noptima, run_statistics
from elite_selection import tournament
from fitness import fitness
from generate_offspring import generate_offspring
from initiate import initiate_populations
from feasibility import feasibility 


#%%

class Problem:
    def __init__(
            self,
            objective, # at the moment this is the optimal cost, later it will be objective function for co-optimisation
            optimum, # at the moment this is the coordinates of optimum cost, later it will be dynamically updated during co-optimisation
            bounds, # [lb[:], ub[:]]

            # *args, 
            
            # vectorized = False, # whether fitness function accepts vectorsied array of points
            
            # **kwargs,
            ):
        
        
        # if vectorized is True:
        #     self.func = fitness
        # else: 
        #     def fitness_wrapper(xn, *args, **kwargs):
        #         """
        #         wrapper for fitness function to allow passing 2d array of points
                
        #         xn.shape = (ndim, ...)
        #         """
        #         result = np.empty(xn.shape[1])
        #         for i, x in enumerate(xn):
        #             result[i] = fitness(x, *args, **kwargs)
        #         return result
        #     self.func = fitness_wrapper
                
        self.objective = objective
        
        # `bounds` is a tuple containing (lb, ub)
        self.lb, self.ub = self.bounds = bounds 
        assert utils.islistlike(self.lb)
        assert utils.islistlike(self.ub)
        assert len(self.lb) == len(self.ub)
        self.ndim = len(self.lb)
        
        
    def Initiate(
            self, 
            nniche: int, 
            maxpop: int, 
            # x0 = None,
            ):
        self.nniche = nniche
        self.maxpop = maxpop

        # self.population is 3d array of shape (nniche, maxpop, ndim)        
        self.population = initiate_populations(self.nniche, self.maxpop, *self.bounds, )
        # assert (self.population.max(axis=(0,1)) <= self.ub).all()
        # assert (self.population.min(axis=(0,1)) >= self.lb).all()
        
        
    def Step(
            self, 
            maxiter: int|float = np.inf,
            maxpop: int = 100, 
            nelite: int = 10,
            tournsize: int = 4,
            mutation: float|tuple[float] = (0.5, 1.5), 
            gaussian_mu: float = 0.0, 
            gaussian_sigma: float = 0.1, 
            crossover: float|tuple[float] = 0.4, 
            slack: float = np.inf,
            
            ):
        """
        Options: 
            maxiter   - maximum number of iterations allowed for the step
            maxpop    - maximum number of individuals in a niche
            nelite    - number of elites kept 
                            (passed on to deap.toolbox.selTournament as k)
            tournsize - number of individuals in each tournament 
                            (passed on to deap.toolbox.selTournament as tournsize)
            slack     - near-optimal fitness relaxation in range (1, np.inf)
        """
        
        # print(kwargs.items())
        self.maxiter = Maxiter(maxiter)
        self.maxpop = maxpop
        self.nelite = nelite
        self.tournsize = tournsize
        self.mutation = mutation
        self.gaussian_sigma = list(gaussian_sigma 
                                   * (self.ub - self.lb)) # adjust sigma for unnormalised bounds
        self.crossover = crossover
        self.slack = slack
        
        while self.maxiter(): 
            self.Loop()
        
        self.Terminate()
        
    def Loop(self):
        
        if hasattr(self.mutation, "__iter__"):
            assert len(self.mutation) == 2
            self.mutationInst = np.random.uniform(*self.mutation)
        else: 
            assert isinstance(self.mutation, float)
            self.mutationInst = self.mutation
        if hasattr(self.crossover, "__iter__"):
            assert len(self.crossover) == 2
            self.crossoverInst = np.random.uniform(*self.crossover)
        else: 
            assert isinstance(self.crossover, float)
            self.crossoverInst = self.crossover
        
        
        self.fitness = feasibility(
            self.population
            )
        
        # select parents
        self.population = tournament(
            self.population, 
            self.nelite, 
            self.tournsize, 
            )
        
        # generate offspring
        self.population = generate_offspring(
            self.population, 
            self.nelite, 
            self.maxpop,
            self.mutationInst,
            self.gaussian_sigma,
            self.crossoverInst,
            )
        
        self.population = feasibility(
            self.population, 
            self.slack
            )
        
        self.fitness = fitness(
            self.population, 
            )

    def Terminate(self):
        run_statistics(self.population)
        self.noptima = return_noptima(self.population)
        return self.noptima
        
#%% 


    
if __name__ == "__main__":
    
    def Objective(x):
        return sum(np.arange(len(x))*x)
    
    lb = 1*np.ones(3)
    ub = 3*np.ones(3)
    objective = Objective(lb)
    
    problem = Problem(
        objective = objective,
        optimum = lb,
        bounds = (lb, ub),
        )
    
    problem.Initiate(4, 10)
    
    raise KeyboardInterrupt
    problem.Step(
        maxiter=10, 
        maxpop=10, 
        nelite=4,
        tournsize=2,
        slack=np.inf,
        )
