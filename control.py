# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:04:52 2025

@author: hmtha
"""

import numpy as np 
from numba import njit

import utils 

from termination_criteria import Maxiter, Convergence

#%%

from deap import base
from deap import creator
from deap import tools

# For now, we only initialise the co-evolving near-optimal niches. 
# The optimum is given as a starting point. In principle, the optimum could be 
# a niche in itself, with its specific fitness function, and become part
# of the same meta-heuristic search as the one that looks for the near-optimal solutions.

# To initialise niches, we first need to create a DEAP fitness object to assign to it
creator.create("Fitness", base.Fitness, weights=(1.0,)) 

# Hence, we can create a list object for individuals to be evaluated based on the above fitness
creator.create(
    "Individual", 
    list, 
    fitness=creator.Fitness, 
    # objective=np.inf,
    # feasibility=False,
    ) 

# We create a toolbox to register attributes we want the individuals to have and
# create the actual instances of the individuals to be packaged into niches
toolbox = base.Toolbox()

#%%

class Problem:
    def __init__(
            self,
            func,
            objective, # at the moment this is the optimal cost, later it will be objective function for co-optimisation
            optimum, # at the moment this is the coordinates of optimum cost, later it will be dynamically updated during co-optimisation
            bounds, # [lb[:], ub[:]]
            
            nniche: int, 
            maxpop: int,
            x0 = np.random.uniform, # callable|np.ndarray
            # vectorized = False, # whether fitness function accepts vectorsied array of points
            
            # **kwargs,
            ):
        
        
        self.func = func
        self.objective = objective
        
        # `bounds` is a tuple containing (lb, ub)
        self.lb, self.ub = self.bounds = bounds 
        assert utils.islistlike(self.lb)
        assert utils.islistlike(self.ub)
        assert len(self.lb) == len(self.ub)
        self.ndim = len(self.lb)
        
        self.nniche = nniche
        self.maxpop = maxpop
        self.noptimal_threshold = np.inf

        ## Initiate population
        # self.population is list of lists like a 3d array of shape (nniche, maxpop, ndim)        
        self._initiate_population(x0)
        
        assert (np.array(self.population).max(axis=(0,1)) <= self.ub).all(), "initial population exceeds bounds"
        assert (np.array(self.population).min(axis=(0,1)) >= self.lb).all(), "initial population exceeds bounds"
        
        # evaluate fitness of initial population
        self._fitness()
    
    def _initiate_population(self, x0):
        if hasattr(x0, "__call__"):
            def create_individual():
                ind = creator.Individual(x0(lb, ub).tolist()) # Create an instance of creator.Individual 
                return ind
        
            toolbox.register("individual", create_individual)
            
            # A 'niche' is a list of the created individuals with the desired attributes
            # Create a list of niches (as many as the desired alternatives)
            self.population = [tools.initRepeat(list, toolbox.individual, n=self.maxpop) 
                               for _ in range(self.nniche)]

        else:
            assert isinstance(x0, np.ndarray)
            assert x0.shape[0] == self.nniche
            assert x0.shape[1] == self.maxpop
            assert x0.shape[2] == self.ndim
                
            self.population = [[creator.Individual(x0[i, j]) 
                                for j in range(self.maxpop)] 
                               for i in range(self.nniche)]
        
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
        
        self.noptimal_threshold = self.objective*self.slack
        
        assert self.elitek + self.tournk < maxpop, "elitek + tournk should be less than maxpop"
        assert self.tournk > self.tournsize, "tournsize should be less than tournk"
        
        while self.maxiter(): 
            self.Loop()
        

    def Loop(self):
        
        if hasattr(self.mutation, "__iter__"):
            self.mutationInst = _dither(*self.mutation)
        else: 
            assert isinstance(self.mutation, float)
            self.mutationInst = self.mutation
        if hasattr(self.crossover, "__iter__"):
            self.crossoverInst = _dither(*self.crossover)
        else: 
            assert isinstance(self.crossover, float)
            self.crossoverInst = self.crossover
        
        # select parents
        self._select_parents()

        # generate offspring
        self._generate_offspring()
        
        # evaluate fitness
        self._fitness()
        
    def Terminate(self):
        self._run_statistics()
        self._return_noptima()
        return self.noptima, self.fitnesses

    def _return_noptima(self):
        self.noptima = [tools.selBest(niche, 1)[0] for niche in self.population]
        self.fitnesses = [noptimum.fitness for noptimum in self.noptima]
    
    def _run_statistics(self):
        """TODO"""
        pass

    def _select_parents(self):
        # order irrelevant
        self.population = [
            tools.selBest(niche, k=self.elitek) + 
            tools.selTournament(niche, k=self.tournk, tournsize=self.tournsize) 
            for niche in self.population]
            
    def _generate_offspring(self):
        nparents = len(self.population[0])
        # clone parents as the deap functions work in-place
        self.population = [[toolbox.clone(niche[idx%nparents]) for idx in range(self.maxpop)] for niche in self.population]
        
        for niche in self.population:
            # shuffle parents for mating
            np.random.shuffle(niche)
            
            for ind1, ind2 in zip(niche[::2], niche[1::2]):
                # crossover probability
                if np.random.random() < self.crossoverInst:
                    # Apply crossover
                    # mate
                    ## cxOnePoint used for testing 
                    ## TODO: choose cx method
                    tools.cxOnePoint(ind1, ind2)
        
            for ind in niche:
                # Apply mutation
                tools.mutGaussian(
                    ind, 
                    mu=0.0, 
                    sigma=self.sigma, 
                    indpb=self.mutationInst
                    )
                apply_bounds(ind)
                # Mark children's fitness as invalid (were retained in cloning)
                del ind.fitness.values
                ind.fitness.valid = False
                
    def _fitness(self):
        """
        Calculates the fitness of each individual as the minimum (per-variable)
        distance from the centroids of all the other niches besides the one 
        where the individual itself sits, and including the 'optimal' niche. 
        """
        centroids = find_centroids(np.array(self.population))
        
        for n, niche in enumerate(self.population):
            for ind in niche: 
                ind.fitness = min((np.abs(np.array(ind) - centroid).min() 
                                   for c, centroid in enumerate(centroids)
                                   if n != c))
                
                ##TODO: 
    # =============================================================================
    #             We want the cost penalty to be similar in scale to the fitnesses 
    #               Need an adjsutment 
    # =============================================================================
                cost = self.func(np.array(ind))
                if cost > self.noptimal_threshold:
                    ind.fitness += cost 
                    

@njit
def find_centroids(population):
    centroids = np.empty((population.shape[0], population.shape[2]))

    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                centroids[i, k] += population[i, j, k]
    centroids /= population.shape[1]

    return centroids       

@njit
def _dither(lower, upper, distribution=np.random.uniform):
    return distribution(lower, upper)

# this cannot be njitted as deap.individual is not compatible
def apply_bounds(ind, lb, ub):
    for i in range(len(ind)):
        ind[i] = min(ub[i], max(lb[i], ind[i]))
    return ind
#%% 


    
if __name__ == "__main__":
    
    def Objective(values): 
        
        # For the first test, ndim=2 and works for a function with two decision variables
        z = sum(np.sin(19*np.pi*values[i]) + values[i]/1.7 for i in range(len(values))) + 2
        
        return z
    
    lb = -1*np.ones(2)
    ub = 0*np.ones(2)
    objective = Objective(lb)
    
    problem = Problem(
        func=Objective,
        objective = -5.146,
        optimum = lb,
        bounds = (lb, ub),
        nniche = 4, 
        maxpop = 100,
        )
    
    problem.Step(
        maxiter=100, 
        maxpop=100, 
        elitek=8,
        tournk=8,
        tournsize=4,
        mutation=0.05, 
        sigma=0.2,
        crossover=0.6, 
        slack=1.12,
        )
    noptima, fitnesses = problem.Terminate()
    for x in noptima: print(x)