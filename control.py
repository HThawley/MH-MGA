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
            bounds, # [lb[:], ub[:]]

            *args, 
            
            # vectorized = False, # whether fitness function accepts vectorsied array of points
            
            **kwargs,
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
        self.lb, self.ub = bounds 
        assert utils.islistlike(self.lb)
        assert utils.islistlike(self.ub)
        assert len(self.lb) == len(self.ub)
        self.ndim = len(self.lb)
        
        
    def Initiate(
            self, 
            nniche: int, 
            maxpop: int, 
            
            *args, 
            x0 = None,
            **kwargs,
            ):
        self.nniche = nniche
        self.maxpop = maxpop

        # self.population is 3d array of shape (nniche, maxpop, ndim)        
        self.population = initiate_populations(self.nniche, self.maxpop, *self.bounds, )
        assert (self.population.max(axis=2) <= self.ub).all()
        assert (self.population.max(axis=2) >= self.lb).all()
        
        
    def Step(
            self, 
            maxiter: int|float = np.inf,
            maxpop: int = 100, 
            nelite: int = 10,
            tournsize: int = 4,
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
        self.slack = slack
        
        while self.maxiter(): 
            self.Loop()
        
        self.Terminate()
        
    def Loop(self):
        
        # select parents
        self.population = tournament(
            self.population, 
            self.nelite, 
            self.tournsize, 
            )
        
        # generate offspring
        self.population = generate_offspring(
            self.population, 
            
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
        
    problem = Problem(
        objective = 9.,
        fitness = fitness, 
        bounds = (np.zeros(3), np.ones(3)),
        vectorized = False,
        
        )

