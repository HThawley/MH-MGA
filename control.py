# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:04:52 2025

@author: hmtha
"""

import numpy as np 
import pandas as pd 

import utils 

from crossover_mutation import crossover, mutation
from termination_criteria import Maxiter, Convergence
from post_process import return_noptima, run_statistics
from elite_selection import tournament
from fitness import fitness
from generate_offspring import generate_offspring
from initiate import initiate_populations

#%%

class Problem:
    def __init__(
            self,
            fitness, # fitness function
            bounds, # [lb[:], ub[:]]

            *args, 
            
            vectorized = False, # whether fitness function accepts vectorsied array of points
            fitness_args = (),
            fitness_kwargs = {}, 
            
            **kwargs,
            ):
        
        # fitness function
        assert utils.isfunclike(fitness)
        
        if vectorized is True:
            self.func = fitness
        else: 
            def fitness_wrapper(xn, *args, **kwargs):
                """
                wrapper for fitness function to allow passing 2d array of points
                
                xn.shape = (ndim, ...)
                """
                result = np.empty(xn.shape[1])
                for i, x in enumerate(xn):
                    result[i] = fitness(x, *args, **kwargs)
                return result
            self.func = fitness_wrapper
                
        # `bounds` is a tuple containing (lb, ub)
        # lb
        self.lb, self.ub = bounds 
        assert utils.islistlike(self.lb)
        assert utils.islistlike(self.ub)
        assert len(self.lb) == len(self.ub)
        self.ndim = len(self.lb)
        

            
        
        
    def Initiate(
            self, 
            n_niche, 
            *args, 
            x0 = None,
            **kwargs,
            ):
        
        self.population = initiate_populations(x0, n_niche)
        
        
        
    def Step(self, **kwargs):
        """
        Options: 
            maxiter - maximum number of iterations allowed for the step
            maxpop  - maximum number of individuals in a niche
            max
        """
        
        # print(kwargs.items())

        if "maxiter" in kwargs.keys():
            assert isinstance(kwargs["maxiter"], int)
            self.maxiter = Maxiter(kwargs["maxiter"])
        else:
            self.maxiter = Maxiter(np.inf)
        
        if "maxpop" in kwargs.keys():
            assert isinstance(kwargs["maxpop"], int)
            self.maxpop = kwargs["maxpop"]
        else: 
            
        
        if "slack" in kwargs.keys():
            assert isinstance(kwargs["slack"], float)
            self.slack = kwargs["slack"]
        else: 
            self.slack = np.inf
        
        while self.maxiter(): 
            
            self.Loop()
        
    def Loop(self):
        
        
        
#%% 
if __name__ == "__main__":
    def add(*args):
        return sum(args)
        
        
    problem = Problem(
        fitness = add, 
        bounds = (np.zeros(3), np.ones(3)),
        
        vectorized = False,
        )

