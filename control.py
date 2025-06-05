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
        self.centre = np.mean(self.ub, self.lb)
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
            self.population = np.empty((self.nniche, self.maxpop, self.ndim))
            for i in range(self.nniche):
                for j in range(self.maxpop):
                    self.population[i, j, :] = x0(self.lb, self.ub)

        else:
            assert isinstance(x0, np.ndarray)
            assert x0.shape[0] == self.nniche
            assert x0.shape[1] == self.maxpop
            assert x0.shape[2] == self.ndim
                
            self.population = x0.copy()
        
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
        return self.noptima, self.nfitnesses

    def _return_noptima(self):
        self.noptima = [selBest(niche, 1)[0] for niche in self.population]
        self.nfitnesses = [self.fitnesses.min() for niche in self.population]
    
    def _run_statistics(self):
        """TODO"""
        pass

    def _select_parents(self):
        self.parents = np.dstack(
            tuple(np.vstack(
                (selBest(self.population[n], self.fitness[n], self.elitek), 
                 selTournament(self.population[n], self.fitness[n], self.tournk, self.tournsize))
                ).T
             for n in range(self.nniche)
             )
            ).T
        
        
    def _generate_offspring(self):
        nparents = self.parents.shape[1]
        # clone parents to make a new population
        self.population = np.empty((self.nniche, self.maxpop, self.ndim))
        for i in range(self.nniche):
            for j in range(self.maxpop):
                for k in range(self.ndim):
                    self.population[i, j, k] = self.parents[i, j%nparents, k]
        
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
                    cxOnePoint(ind1, ind2)
        
            for ind in niche:
                # Apply mutation
                mutGaussian(
                    ind, 
                    mu=self.centre, 
                    sigma=self.sigma, 
                    indpb=self.mutationInst
                    )
                apply_bounds(ind)
                # Mark children's fitness as invalid (were retained in cloning)
                
    def _fitness(self):
        """
        Calculates the fitness of each individual as the minimum (per-variable)
        distance from the centroids of all the other niches besides the one 
        where the individual itself sits, and including the 'optimal' niche. 
        """
        self.centroids = find_centroids(np.array(self.population))
        self.fitness = np.empty((self.population.shape[0], 
                                 self.population.shape[1]))
        
        for n in range(self.nniche):
            for i in range(self.maxpop): 
                self.fitness[n, i] = min((np.abs(self.population[n, i] - self.centroids[c]).min() 
                                          for c in range(self.nniche)
                                          if n != c))
                
                ##TODO: 
    # =============================================================================
    #             We want the cost penalty to be similar in scale to the fitnesses 
    #               Need an adjsutment 
    # =============================================================================
                cost = self.func(self.population[n, i])
                if cost > self.noptimal_threshold:
                    self.fitness[n, i] += cost 
        
   
@njit
def mutGaussian(ind, mu, sigma, indpb):
    """Gaussian mutation for NumPy arrays."""
    for i in range(len(ind)):
        if np.random.random() < indpb:
            ind[i] += np.random.normal(mu[i], sigma[i])
    return ind
          
@njit
def cxOnePoint(ind1, ind2):
    index1 = np.random.randint(0, len(ind1))
    index2 = len(ind1)
    _do_cx(ind1, ind2, index1, index2)
    
@njit
def cxTwoPoint(ind1, ind2):
    # +1 / -1 adjustments are made to ensure there is always a crossover 
    # only valid for ndim >= 3
    index1 = np.random.randint(0, len(ind1)-1)
    index2 = np.random.randint(index1+1, len(ind1))
    _do_cx(ind1, ind2, index1, index2)

@njit
def _do_cx(ind1, ind2, index1, index2):
    buffer = 0.0
    # modify in place
    for i in range(index1, index2):
        buffer = ind1[i]
        ind1[i] = ind2[i]
        ind2[i] = buffer 
        

@njit
def selTournament(niche, fitnesses, n, tournsize):
    
    selected = np.empty((n, niche.shape[1]))
    selected_fitness = np.empty(n)
    
    indices = np.empty(tournsize, np.int64)
    for m in range(n):
        indices[:] = np.random.randint(0, niche.shape[0], size=tournsize)
            
        selected_index = indices[fitnesses[indices].argmin()]
        selected_fitness[m] = fitnesses[selected_index]
        selected[m, :] = niche[selected_index, :]
    
    return selected # , selected_fitness
    
@njit
def selBest(niche, fitnesses, n):
    # holds coordinates of best n individuals (in order)
    selected = np.empty((n, niche.shape[1]))
    # holds fitnesses of best n individuals (in order)
    selected_fitness = np.full(n, np.inf, np.float64)
    # holds indices of best n individuals (in order)
    indices = np.empty(n, np.int64)
    
    # loop through individuals
    for i in range(niche.shape[0]):
        # loop through n
        for j in range(n):
            # if individual is better than the current jth best
            if fitnesses[i] < selected_fitness[j]:
                # move each current elite up the chain, losing the nth
                for k in range(n-1, j-1, -1):
                    selected_fitness[k] = selected_fitness[k-1]
                    indices[k] = indices[k-1]
                # add the current individual 
                selected_fitness[j] = fitnesses[i]
                indices[j] = i
                break
    # coordinates
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
    
    return selected # , selected_fitness
    

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

# this cannot be njitted as individual is not compatible
@njit
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