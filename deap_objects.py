# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:22:59 2025

@author: hmtha
"""

from deap import base
from deap import creator

# For now, we only initialise the co-evolving near-optimal niches. 
# The optimum is given as a starting point. In principle, the optimum could be 
# a niche in itself, with its specific fitness function, and become part
# of the same meta-heuristic search as the one that looks for the near-optimal solutions.

# To initialise niches, we first need to create a DEAP fitness object to assign to it
creator.create("Fitness", base.Fitness, weights=(1.0,)) 

# Hence, we can create a list object for individuals to be evaluated based on the above fitness
creator.create("Individual", list, fitness=creator.Fitness) 

# We create a toolbox to register attributes we want the individuals to have and
# create the actual instances of the individuals to be packaged into niches
toolbox = base.Toolbox()