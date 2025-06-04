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
from elite_selection import select_parents
from generate_offspring import generate_offspring
from initiate import initiate_populations
from feasibility import fitness 


#%%

class Problem:
    def __init__(
            self,
            func,
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
        self.func = func
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

        # self.population is list of lists like a 3d array of shape (nniche, maxpop, ndim)        
        self.population = initiate_populations(self.nniche, self.maxpop, *self.bounds, )
        # assert (self.population.max(axis=(0,1)) <= self.ub).all()
        # assert (self.population.min(axis=(0,1)) >= self.lb).all()
        
        self.population = fitness(
            self.population,
            self.func, 
            self.slack, 
            self.objective,
            )
        
    def Step(
            self, 
            maxiter: int|float = np.inf, # max no of iterations in this step
            maxpop: int = 100, # max no of individuals in each niche
            elitek: int = 5, # number of parents selected as absolute best
            tournk: int = 10, # number of parents selected via tournament
            tournsize: int = 4, # tournament size parameter
            mutation: float|tuple[float] = (0.5, 1.5), # mutation probability
            sigma: float = 0.1, # standard deviation of gaussian noise for mutation (gets scaled)
            crossover: float|tuple[float] = 0.4, # crossover probability
            slack: float = np.inf, # noptimal slack in range (1.0, inf)
            
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
        
        self.maxiter = Maxiter(maxiter)
        self.maxpop = maxpop
        self.elitek = elitek
        self.tournk = tournk
        self.tournsize = tournsize
        self.mutation = mutation
        self.sigma = list(sigma 
                          * (self.ub - self.lb)) # adjust sigma for unnormalised bounds
        self.crossover = crossover
        self.slack = slack
        
        assert self.elitek + self.tournk < maxpop, "elitek + tournk should be less than maxpop"
        assert self.tournk < self.tournsize, "tournsize should be less than tournk"
        
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
        
        
        # select parents
        self.population = select_parents(
            self.population, 
            self.elitek, 
            self.tournk, 
            self.tournsize, 
            )

        # generate offspring
        self.population = generate_offspring(
            self.population, 
            self.elitek + self.tournk, 
            self.maxpop,
            self.mutationInst,
            self.sigma,
            self.crossoverInst,
            )
        
        self.population = fitness(
            self.population,
            self.func, 
            self.slack, 
            self.objective,
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
        elitek=4,
        tournk=4,
        tournsize=2,
        mutation=(0.5, 1.5), 
        sigma=0.1,
        crossover=0.5, 
        slack=np.inf,
        )
