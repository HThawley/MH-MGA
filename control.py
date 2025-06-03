# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:04:52 2025

@author: hmtha
"""

import numpy as np 
import pandas as pd 

import utils 

from crossover_mutation import crossover, mutation
from termination_criteria import maxiter, convergence
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
            
            ### TODO: 
            # x0 = None, # starting point(s)
            # vectorized = False, # whether fitness function accepts vectorsied array of points

            **kwargs,
            ):
        
        # fitness function
        self.fitness = fitness 
        assert utils.isfunclike(self.fitness)        
        
        # `bounds` is a tuple containing (lb, ub)
        # lb
        self.lb, self.ub = bounds 
        assert utils.islistlike(self.lb)
        assert utils.islistlike(self.ub)
        assert len(self.lb) == len(self.ub)
        self.ndim = len(self.lb)
        
    
        
        


