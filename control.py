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
            bounds, # [lb[:], ub[:]]
            
            nniche: int, 
            popsize: int,
            x0 = np.random.uniform, # callable|np.ndarray
            maximize = False,
            # vectorized = False, # whether fitness function accepts vectorsied array of points
            
            # **kwargs,
            ):
        
        self.func = func

        assert isinstance(maximize, bool)
        self.maximize = maximize
        
        # `bounds` is a tuple containing (lb, ub)
        self.lb, self.ub = self.bounds = bounds 
        self.centre = (self.ub - self.lb) / 2 + self.lb
        self.optimum = self.centre.copy()
        self.optimal_obj = func(self.optimum)
        
        assert utils.islistlike(self.lb)
        assert utils.islistlike(self.ub)
        assert len(self.lb) == len(self.ub)
        self.ndim = len(self.lb)
        
        self.nniche = nniche
        self.popsize = popsize
        self.noptimal_threshold = -np.inf if self.maximize else np.inf

        ## Initiate population
        # self.population is list of lists like a 3d array of shape (nniche, popsize, ndim)        
        self.Initiate_population(x0)
        
        assert (np.array(self.population).max(axis=(0,1)) <= self.ub).all(), "initial population exceeds bounds"
        assert (np.array(self.population).min(axis=(0,1)) >= self.lb).all(), "initial population exceeds bounds"
        
        # evaluate fitness of initial population
        self.Evaluate_objective()
        self.Update_optimum()
        self.Evaluate_feasibility()
        self.Evaluate_fitness()
    
    def Initiate_population(self, x0):
        if hasattr(x0, "__call__"):
            self.population = np.empty((self.nniche, self.popsize, self.ndim))
            for i in range(self.nniche):
                for j in range(self.popsize):
                    self.population[i, j, :] = x0(self.lb, self.ub)

        else:
            assert isinstance(x0, np.ndarray)
            assert x0.shape[0] == self.nniche
            assert x0.shape[1] == self.popsize
            assert x0.shape[2] == self.ndim
                
            self.population = x0.copy()
        
    def Step(
            self, 
            maxiter: int|float = np.inf, # max no of iterations in this step
            popsize: int = 100, # max no of individuals in each niche
            elitek: int|float = 0.1, # number of parents selected as absolute best
            tournk: int|float = 0.9, # number of parents selected via tournament
            tournsize: int = 4, # tournament size parameter
            mutation: float|tuple[float] = (0.5, 1.5), # mutation probability
            sigma: float = 0.1, # standard deviation of gaussian noise for mutation (gets scaled)
            crossover: float|tuple[float] = 0.4, # crossover probability
            slack: float = np.inf, # noptimal slack in range (1.0, inf)
            ):
        
        self.maxiter = Maxiter(maxiter)
        self.popsize = popsize
        
        assert elitek!=-1 or tournk !=-1, "only 1 of elitek and tournk may be -1"
        self.elitek = elitek if isinstance(elitek, int) else int(elitek*self.popsize)
        self.tournk = tournk if isinstance(tournk, int) else int(tournk*self.popsize)
        self.elitek = self.popsize - self.tournk if self.elitek == -1 else self.elitek
        self.tournk = self.popsize - self.elitek if self.tournk == -1 else self.tournk

        self.tournsize = tournsize
        self.mutation = mutation
        self.sigma = sigma * (self.ub - self.lb) # adjust sigma for unnormalised bounds
        self.crossover = crossover
        self.slack = slack
        
        if self.maximize:
            self.noptimal_threshold = self.optimal_obj * (1-(self.slack - 1))
        else: 
            self.noptimal_threshold = self.optimal_obj*self.slack
        
        assert self.elitek + self.tournk <= popsize, "elitek + tournk should be weakly less than popsize"
        assert self.popsize > self.tournsize, "tournsize should be less than popsize"
        
        while not self.maxiter(): 
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
        self.Select_parents()
        # generate offspring
        self.Generate_offspring()
        # evaluate objective
        self.Evaluate_objective()
        # update optimum
        self.Update_optimum()
        # update feasibility
        self.Evaluate_feasibility()
        # update fitness
        self.Evaluate_fitness()
        
    def Terminate(self):
        self.Run_statistics()
        self.Return_noptima()
        return self.noptima, self.nfitness

    def Update_optimum(self):
        if self.cost.max() < self.optimal_obj:
            return 
        else: 
            for n in range(self.nniche):
                if self.cost[n].max() < self.optimal_obj:
                    continue
                self.optimal_obj = self.cost[n].max()
                self.optimum = self.population[n, self.cost[n].argmax(), :]

    def Return_noptima(self):
        self.noptima = [selBest(
            self.population[n], 
            self.fitness[n], 
            self.feasibility[n], 
            self.cost[n],
            1, 
            self.maximize
            )[0] for n in range(self.nniche)]
        self.nfitness = [self.fitness[n].max() for n in range(self.nniche)]
    
    def Run_statistics(self):
        """TODO"""
        pass

    def Select_parents(self):
        self.parents = np.dstack(
            (np.vstack(
                (selBest(
                    self.population[0], 
                    self.cost[0], 
                    self.feasibility[0], 
                    self.cost[0],
                    self.elitek, 
                    self.maximize
                     ), 
                 selTournament(
                     self.population[0], 
                     self.cost[0], 
                     self.feasibility[0], 
                     self.cost[0],
                     self.tournk, 
                     self.tournsize, 
                     self.maximize
                     )
                 )
                ).T,
                )
            +
            tuple(np.vstack(
                (selBest(
                    self.population[n], 
                    self.fitness[n], 
                    self.feasibility[n], 
                    self.cost[n],
                    self.elitek, 
                    self.maximize
                    ), 
                 selTournament(
                     self.population[n], 
                     self.fitness[n], 
                     self.feasibility[n], 
                     self.cost[n],
                     self.tournk, 
                     self.tournsize, 
                     self.maximize
                     )
                 )
                ).T
             for n in range(1, self.nniche)
             )
            ).T
        
        
    def Generate_offspring(self):
        nparents = self.parents.shape[1]
        # clone parents to make a new population
        self.population = np.empty((self.nniche, self.popsize, self.ndim))
        for i in range(self.nniche):
            for j in range(self.popsize):
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
                    sigma=self.sigma, 
                    indpb=self.mutationInst
                    )
                apply_bounds(ind, self.lb, self.ub)
                # Mark children's fitness as invalid (were retained in cloning)
                
    def Evaluate_fitness(self):
        """
        Calculates the fitness of each individual as the minimum (per-variable)
        distance from the centroids of all the other niches besides the one 
        where the individual itself sits, and including the 'optimal' niche. 
        """
        self.centroids = find_centroids(np.array(self.population))
        self.fitness = np.empty((self.population.shape[0], 
                                 self.population.shape[1]))
        
        for n in range(self.nniche):
            for i in range(self.popsize): 
                
                self.fitness[n, i] = min((euclideanDistance(self.population[n, i], 
                                                            self.centroids[c]) for c in range(self.nniche)
                                                           if n != c))
                
                ##TODO: 
                # =============================================================================
                #   We want the cost penalty to be similar in scale to the fitness 
                #       Need an adjsutment 
                # =============================================================================
                if not self.feasibility[n, i]:
                    if self.maximize: 
                        self.fitness[n, i] -= abs(self.cost[n, i])*1000 
                    else:
                        self.fitness[n, i] += abs(self.cost[n, i])*1000

    def Evaluate_objective(self):
        """
        Calculates the objective function for each individual
        """
        self.cost = np.empty((self.population.shape[0], 
                              self.population.shape[1]))
        for n in range(self.nniche):
            for i in range(self.popsize):
                self.cost[n, i] = self.func(self.population[n, i])
                ##TODO: what if cost crosses +/- ???
                    # 0 would be a stable point which is bad
                        
    def Evaluate_feasibility(self):
        self.feasibility = np.ones((self.population.shape[0], 
                                    self.population.shape[1]), 
                                   dtype=np.bool_)
        for n in range(self.nniche):
            for i in range(self.popsize):
                ##TODO: what if cost crosses +/- ???
                    # 0 would be a stable point which is bad
                if self.maximize: 
                    if self.cost[n, i] < self.noptimal_threshold:
                        self.feasibility[n, i] = False
                else: 
                    if self.cost[n, i] > self.noptimal_threshold:
                        self.feasibility[n, i] = False
        

        
@njit
def euclideanDistance(p1, p2):
    """Euclidean distance"""
    return sum((p1-p2)**2)**0.5
    
@njit
def pointwiseDistance(p1, p2):
    return min(np.abs(p1-p2))
    
@njit
def mutGaussian(ind, sigma, indpb):
    """Gaussian mutation for NumPy arrays."""
    for i in range(len(ind)):
        if np.random.random() < indpb:
            ind[i] += np.random.normal(ind[i], sigma[i])
    return ind

@njit
def _do_cx(ind1, ind2, index1, index2):
    buffer = 0.0
    # modify in place
    for i in range(index1, index2):
        buffer = ind1[i]
        ind1[i] = ind2[i]
        ind2[i] = buffer   

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
def selTournament(niche, fitness, feasibility, cost, n, tournsize, maximize):
    
    selected = np.empty((n, niche.shape[1]))
    selected_value = np.empty(n)
    
    indices = np.empty(tournsize, np.int64)
    if maximize:
        for m in range(n):            
            indices[:] = np.random.randint(0, niche.shape[0], size=tournsize)
                
            if feasibility[indices].sum() < tournsize / 2: # majority infeasible
                selection = cost
            else: 
                selection = fitness
            
            selected_index = indices[selection[indices].argmax()]
            selected_value[m] = selection[selected_index]
            selected[m, :] = niche[selected_index, :]
    else:
        for m in range(n):
            indices[:] = np.random.randint(0, niche.shape[0], size=tournsize)
                
            if feasibility[indices].sum() < tournsize / 2: # majority infeasible
                selection = cost
            else: 
                selection = fitness
            
            selected_index = indices[selection[indices].argmin()]
            selected_value[m] = selection[selected_index]
            selected[m, :] = niche[selected_index, :]
    
    return selected # , selected_value
    
@njit
def selBest(niche, fitness, feasibility, cost, n, maximize):
    # holds coordinates of best n individuals (in order)
    selected = np.empty((n, niche.shape[1]))
    fill = -np.inf if maximize else np.inf
    # holds fitness of best n individuals (in order)
    selected_value = np.full(n, fill, np.float64)
    # holds indices of best n individuals (in order)
    indices = np.empty(n, np.int64)
    
    if feasibility.sum() < len(feasibility) / 2: # majority infeasible
        selection = cost
    else: 
        selection = fitness
    
    if maximize:
        # loop through individuals
        for i in range(niche.shape[0]):
            # loop through n
            for j in range(n):
                # if individual is better than the current jth best
                if selection[i] > selected_value[j]:
                    # move each current elite up the chain, losing the nth
                    for k in range(n-1, j, -1): 
                        selected_value[k] = selected_value[k-1]
                        indices[k] = indices[k-1]
                    # add the current individual 
                    selected_value[j] = selection[i]
                    indices[j] = i
                    break
    else:
        # loop through individuals
        for i in range(niche.shape[0]):
            # loop through n
            for j in range(n):
                # if individual is better than the current jth best
                if selection[i] < selected_value[j]:
                    # move each current elite up the chain, losing the nth
                    for k in range(n-1, j, -1):
                        selected_value[k] = selected_value[k-1]
                        indices[k] = indices[k-1]
                    # add the current individual 
                    selected_value[j] = selection[i]
                    indices[j] = i
                    break
                
    # coordinates
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
    
    return selected # , selected_value
    

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

@njit
def apply_bounds(ind, lb, ub):
    for i in range(len(ind)):
        ind[i] = max(lb[i], ind[i])
        ind[i] = min(ub[i], ind[i])
    return ind
#%% 


    
if __name__ == "__main__":
    # @njit
    # def Objective(values): 
    #     # For the first test, ndim=2 and works for a function with two decision variables
    #     z = 2.0
    #     for i in range(len(values)):
    #         z += np.sin(19*np.pi*values[i]) + values[i]/1.7
    #     return z
    
    def Objective(values): 
    
        # For the first test, ndim=2 and works for a function with two decision variables
        z = sum(np.sin(19*np.pi*values[i]) + values[i]/1.7 for i in range(len(values))) + 2
        
        return z
    
    # def Objective(x):
    #     return ((0.5*(x-0.55)**2 + 0.9)*np.sin(5*np.pi*x)**6)[0]
    
    lb = 0*np.ones(2)
    ub = 1*np.ones(2)
    
    # alternatives = np.array([[0.021, 0.974],
    #                [0.974, 0.021],
    #                [0.443, 0.549]])
    # x0 = np.dstack(tuple(np.repeat(alternatives[n], 100).reshape(2, 100).T for n in range(3))).transpose(2, 0, 1)
    
    
    
    problem = Problem(
        func=Objective,
        bounds = (lb, ub),
        nniche = 4, 
        popsize = 1000,
        maximize = True, 
        # x0 = x0,
        )

    problem.Step(
        maxiter=100, 
        popsize=1000, 
        elitek=10,
        tournk=-1,
        tournsize=2,
        mutation=0.1, 
        sigma=0.4,
        crossover=0.4, 
        slack=1.12,
        )
    noptima, nfitness = problem.Terminate()
    for n in range(4): print(noptima[n], nfitness[n])
    
# B (0.021, 0.974), C (0.974, 0.021), and D (0.443, 0.549) (or E (0.549, 0.443))