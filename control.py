# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:04:52 2025

@author: hmtha
"""

import numpy as np 
from numba import njit

from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist

import utils 

from termination_criteria import Maxiter, Convergence
from timekeeper import keeptime, PrintTimekeeper, timekeeper

#%%
# =============================================================================
# TODO: update fitness so as not to calculate it for optimum niche
# TODO: Update maximise logic to differentiate maximising fitness and maximising objective
# =============================================================================

class Problem:
    @keeptime("init")
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
        
        assert nniche >= 2
        self.nniche = nniche
        self.popsize = popsize
        self.noptimal_obj = -np.inf if self.maximize else np.inf


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
    
    @keeptime("Instantiate_working_arrays")
    def Instantiate_working_arrays(self):
        """ 
        For fast njit functions it helps to work on arrays in place and therfore
        avoid allocating memory 
        """
        self.objective = np.empty((self.nniche, self.popsize), dtype=np.float64)
        self.feasibility = np.empty((self.nniche, self.popsize), dtype=np.bool_)
        self.fitness = np.empty((self.nniche, self.popsize), dtype=np.float64)
        self.parents = np.empty((self.nniche, self.tournk + self.elitek, self.ndim), 
                                dtype=np.float64)
        self.centroids = np.empty((self.nniche, self.ndim), np.float64)

    @keeptime("Initiate_population")
    def Initiate_population(self, x0):
        self.population = np.empty((self.nniche, self.popsize, self.ndim))
        self.objective = np.empty((self.nniche, self.popsize), dtype=np.float64)
        self.feasibility = np.empty((self.nniche, self.popsize), dtype=np.bool_)
        self.fitness = np.empty((self.nniche, self.popsize), dtype=np.float64)
        self.centroids = np.empty((self.nniche, self.ndim), np.float64)

        if hasattr(x0, "__call__"):
            for i in range(self.nniche):
                for j in range(self.popsize):
                    self.population[i, j, :] = x0(self.lb, self.ub)

        else:
            assert isinstance(x0, np.ndarray)
            assert x0.shape == (self.nniche, self.popsize, self.ndim)
            
            for i in range(self.nniche):
                for j in range(self.popsize):
                    self.population[i, j, :] = x0[i, j]
    
    @keeptime("Step")
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
            new_niche: int = 0, 
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
            self.noptimal_obj = self.optimal_obj * (1-(self.slack - 1))
        else: 
            self.noptimal_obj = self.optimal_obj*self.slack
        
        if new_niche is not None:
            self.Add_niche(new_niche)
        
        assert self.elitek + self.tournk <= popsize, "elitek + tournk should be weakly less than popsize"
        assert self.popsize > self.tournsize, "tournsize should be less than popsize"
        
        self.Instantiate_working_arrays()
        
        while not self.maxiter(): 
            self.Loop()
    
    @keeptime("Add_niche")
    def Add_niche(self, nniche):
        old_pop = self.population.copy()
        self.population = np.empty((self.nniche + nniche, self.popsize, self.ndim))
        for i in range(self.nniche):
            for j in range(self.popsize):
                for k in range(self.ndim):
                    self.population[i, j, k] = old_pop[i, j, k]
        
        if self.nniche < self.ndim + 1 or self.ndim == 1:
            for i in range(self.nniche, self.nniche + nniche):
                for j in range(self.popsize):
                    for k in range(self.ndim):
                        self.population[i, j, k] = np.random.uniform(self.lb[k], self.ub[k])
            
            self.nniche += nniche
            return 
        
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
    
        best = radii.argsort()
        sigma = 1.0*(self.ub-self.lb)
        for i in range(nniche):
            self.population[self.nniche+nniche-1, 0] = candidates[best[-(i+1)]]
            for j in range(1, self.popsize):
                self.population[self.nniche+nniche-1, j] = self.population[self.nniche+nniche-1, 0]
                mutGaussian(self.population[self.nniche+nniche-1, j], sigma, 0.1)
                apply_bounds(self.population[self.nniche+nniche-1, j], self.lb, self.ub)
        self.nniche += nniche
        return 
        
    @keeptime("Loop")        
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
        
        self.Select_parents()
        self.Generate_offspring()
        self.Evaluate_objective()
        self.Update_optimum()
        self.Evaluate_feasibility()
        self.Evaluate_fitness()
        
    @keeptime("Terminate")
    def Terminate(self):
        self.Run_statistics()
        self.Return_noptima()
        return self.noptima, self.nfitness, self.nobjective, self.nfeasibility

    @keeptime("Update_optimum")
    def Update_optimum(self):
    # =============================================================================
    # This can be njitted and refactored for speed up
    # =============================================================================
        if self.maximize:
            if self.objective.max() > self.optimal_obj:
                for n in range(self.nniche):
                    if self.objective[n].max() < self.optimal_obj:
                        continue
                    self.optimal_obj = self.objective[n].max()
                    self.optimum = self.population[n, self.objective[n].argmax(), :]
        else: 
            if self.objective.min() < self.optimal_obj:
                for n in range(self.nniche):
                    if self.objective[n].min() < self.optimal_obj:
                        continue
                    self.optimal_obj = self.objective[n].min()
                    self.optimum = self.population[n, self.objective[n].argmin(), :]

    @keeptime("Return_noptima")
    def Return_noptima(self):
        if self.maximize:
            nindex = [self.objective[0].argmax()]
            nindex += [self.fitness[n].argmax() for n in range(1, self.nniche)]
            
        else:
            nindex = [self.objective[0].argmax()]
            nindex += [self.fitness[n].argmax() for n in range(1, self.nniche)]
            
        self.noptima = [self.population[0, nindex[0]]]
        self.noptima += [self.population[n, nindex[n]] for n in range(1, self.nniche)]
        self.nfitness = [self.fitness[n, nindex[n]] for n in range(self.nniche)]
        self.nobjective = [self.objective[n, nindex[n]] for n in range(self.nniche)]
        self.nfeasibility = [self.feasibility[n, nindex[n]] for n in range(self.nniche)]
        
    
    def Run_statistics(self):
        """TODO"""
        pass

    @keeptime("Select_parents")
    def Select_parents(self):
        _select_parents(
                self.parents, self.population, self.fitness, self.objective, 
                self.feasibility, self.elitek, self.tournk, self.tournsize, self.maximize)
    
    @keeptime("Generate_offspring")
    def Generate_offspring(self):
        # =============================================================================
        # TODO: offload to njit        
        # =============================================================================
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
        self.population[0][0] = self.optimum

    @keeptime("Evaluate_fitness")        
    def Evaluate_fitness(self):
        """
        Calculates the fitness of each individual as the minimum (per-variable)
        distance from the centroids of all the other niches besides the one 
        where the individual itself sits, and including the 'optimal' niche. 
        """
        find_centroids(self.centroids, self.population)
        
        # =============================================================================
        #  TODO: We want the objective penalty to be similar in scale to the fitness 
        #       Need an adjsutment 
        #  TODO: what if objective crosses +/- ??? 0 would be a stable point which is bad
        # =============================================================================
        if self.maximize: 
            _evaluate_fitness_max(
                self.fitness, self.population, self.centroids, self.objective, self.feasibility)
        else:
            _evaluate_fitness_min(
                self.fitness, self.population, self.centroids, self.objective, self.feasibility)

    @keeptime("Evaluate_objective")        
    def Evaluate_objective(self):
        """
        Calculates the objective function for each individual
        """
        for n in range(self.nniche):
            for i in range(self.popsize):
                self.objective[n, i] = self.func(self.population[n, i])

    @keeptime("Evaluate_feasibility")        
    def Evaluate_feasibility(self):
        _evaluate_feasibility(
            self.feasibility, self.population, self.objective, self.noptimal_obj, self.maximize)
    
@njit
def _evaluate_fitness_max(fitness, population, centroids, objective, feasibility):
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
            if not feasibility[i, j]:
                fitness[i, j] -= abs(objective[i, j])*1000 

@njit
def _evaluate_fitness_min(fitness, population, centroids, objective, feasibility):
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            fitness[i, j] = np.inf
            for c in range(population.shape):
                if i == c: 
                    continue
                fitness[i, j] = min(
                    fitness[i, j], 
                    euclideanDistance(population[i, j], centroids[c])
                    )
            if not feasibility[i, j]:
                fitness[i, j] += abs(objective[i, j])*1000 
    
@njit
def _select_parents(
        parents, population, fitness, objective, feasibility, elitek, tournk, tournsize, maximize):
    
    dummy = selBest(population[0, :, :], objective[0, :], elitek)
    for j in range(dummy.shape[0]):
        for k in range(dummy.shape[1]):
            parents[0, j, k] = dummy[j, k]
    dummy = selTournament(
        population[0, :, :], objective[0, :], tournk, tournsize, maximize)
    for j, jp in enumerate(range(elitek, elitek+dummy.shape[0])):
        for k in range(dummy.shape[1]):
            parents[0, jp, k] = dummy[j, k]
    # parents[0, :elitek, :] = selBest(population[0, :, :], objective[0, :], elitek)
    # parents[0, elitek:, :] = selTournament(
    #     population[0, :, :], objective[0, :], tournk, tournsize)
        
    for i in range(1, parents.shape[0]):
        dummy = selBest_fallback(
            population[i, :, :], fitness[i, :], feasibility[i, :], objective[i, :], elitek, maximize)
        for j in range(dummy.shape[0]):
            for k in range(dummy.shape[1]):
                parents[i, j, k] = dummy[j, k]
        dummy = selTournament_fallback(
            population[i, :, :], fitness[i, :], feasibility[i, :], objective[i, :], tournk, tournsize, maximize)
        for j, jp in enumerate(range(elitek, elitek+dummy.shape[0])):
            for k in range(dummy.shape[1]):
                parents[i, jp, k] = dummy[j, k]

        # parents[i, :elitek, :] = selBest_fallback(
        #     population[i, :, :], fitness[i, :], feasibility[i, :], objective[i, :], elitek, maximize)
        # parents[i, elitek:, :] = selTournament_fallback(
        #     population[i, :, :], fitness[i, :], feasibility[i, :], objective[i, :], tournk, tournsize, maximize)
    
@njit
def _evaluate_feasibility(feasibility, population, objective, noptimal_obj, maximize):
    if maximize:
        for i in range(population.shape[0]):
            for j in range(population.shape[1]):
                feasibility[i, j] = objective[i, j] > noptimal_obj
    else: 
        for i in range(population.shape[0]):
            for j in range(population.shape[1]):
                feasibility[i, j] = objective[i, j] < noptimal_obj
    
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

@keeptime("_selT_draw_indices")
@njit 
def _selT_draw_indices(indices, n, size):
    indices[:] = np.random.randint(0, n, size=size)

@keeptime("selTournament_fallback")
@njit
def selTournament_fallback(niche, fitness, feasibility, objective, n, tournsize, maximize):
    """select on fitness preferred. fitness always maximised. 
    objective max/minimized based on value of `maximize`"""
    selected = np.empty((n, niche.shape[1]), np.float64)
    # selected_value = np.empty(n, np.float64)
    indices = np.empty(tournsize, np.int64)
    
    if _sel_fallback(feasibility): # mostly infeasible
        if maximize:
            for m in range(n):
                _selT_draw_indices(indices, niche.shape[0], tournsize)
                selected_idx = 0
                best = -np.inf
                for idx in indices:
                    if objective[idx] > best:
                        selected_idx = idx
                        best = objective[idx]
                # selected_value[m] = best_obj
                selected[m, :] = niche[selected_idx, :]
        else:
            for m in range(n):            
                _selT_draw_indices(indices, niche.shape[0], tournsize)
                selected_idx = 0
                best = np.inf
                for idx in indices:
                    if objective[idx] < best:
                        selected_idx = idx
                        best = objective[idx]
                # selected_value[m] = best_obj
                selected[m, :] = niche[selected_idx, :]
    else:
        for m in range(n):            
            _selT_draw_indices(indices, niche.shape[0], tournsize)
            selected_idx = 0
            best = -np.inf
            for idx in indices:
                if fitness[idx] > best:
                    selected_idx = idx
                    best = fitness[idx]
            # selected_value[m] = best
            selected[m, :] = niche[selected_idx, :]
    
    return selected # , selected_value

@keeptime("selTournament")
@njit
def selTournament(niche, criterion, n, tournsize, maximize):
    """agnostic to criteria"""
    selected = np.empty((n, niche.shape[1]), np.float64)
    # selected_value = np.empty(n, np.float64)
    indices = np.empty(tournsize, np.int64)
    if maximize:
        for m in range(n):            
            _selT_draw_indices(indices, niche.shape[0], tournsize)
            selected_idx = 0
            best = -np.inf
            for idx in indices:
                if criterion[idx] > best:
                    selected_idx = idx
                    best = criterion[idx]
            # selected_value[m] = best
            selected[m, :] = niche[selected_idx, :]
    else:
        for m in range(n):            
            _selT_draw_indices(indices, niche.shape[0], tournsize)
            selected_idx = 0
            best = np.inf
            for idx in indices:
                if criterion[idx] < best:
                    selected_idx = idx
                    best = criterion[idx]
            # selected_value[m] = best
            selected[m, :] = niche[selected_idx, :]

    return selected # , selected_value

@keeptime("_sel_fallback")
@njit
def _sel_fallback(feasibility):
    _feas = 0
    for i in range(len(feasibility)):
        if feasibility[i]: 
            _feas+=1
    _feas /= len(feasibility)
    return _feas < 0.5 

@keeptime("_selBest_chain")
@njit
def _selBest_chain(i, j, n, selected_value, indices, selection):
    for k in range(n-1, j, -1): 
        selected_value[k] = selected_value[k-1]
        indices[k] = indices[k-1]
    selected_value[j] = selection[i]
    indices[j] = i

@keeptime("selBest_fallback")
@njit
def selBest_fallback(niche, fitness, feasibility, objective, n, maximize):
    selected = np.empty((n, niche.shape[1]))
    indices = np.empty(n, np.int64)
    
    if _sel_fallback(feasibility): # mostly infeasible
        fill = -np.inf if maximize else np.inf
        selected_value = np.full(n, fill, np.float64)
        if maximize:
            for i in range(niche.shape[0]):
                for j in range(n):
                    if objective[i] > selected_value[j]:
                        _selBest_chain(i, j, n, selected_value, indices, objective)
                        break
        else:
            for i in range(niche.shape[0]):
                for j in range(n):
                    if objective[i] < selected_value[j]:
                        _selBest_chain(i, j, n, selected_value, indices, objective)
                        break
    else:
        selected_value = np.full(n, -np.inf, np.float64)
        for i in range(niche.shape[0]):
            for j in range(n):
                if fitness[i] > selected_value[j]:
                    _selBest_chain(i, j, n, selected_value, indices, fitness)
                    break
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
    return selected # , selected_value  

@keeptime("selBest")
@njit
def selBest(niche, fitness, n):
    selected = np.empty((n, niche.shape[1]))
    indices = np.empty(n, np.int64)
    selected_value = np.full(n, np.inf, np.float64)
    for i in range(niche.shape[0]):
        for j in range(n):
            if fitness[i] < selected_value[j]:
                _selBest_chain(i, j, n, selected_value, indices, fitness)
                break
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
    return selected # , selected_value

@keeptime("find_centroids")
@njit
def find_centroids(centroids, population):
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                centroids[i, k] += population[i, j, k]
    centroids /= population.shape[1]

@njit
def _dither(lower, upper, distribution=np.random.uniform):
    return distribution(lower, upper)

@keeptime("apply_bounds")
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
        nniche = 3, 
        popsize = 100,
        maximize = True, 
        # x0 = x0,
        )

    problem.Step(
        maxiter=100, 
        popsize=100, 
        elitek=0.2,
        tournk=0.8,
        tournsize=2,
        mutation=0.1, 
        sigma=1.0,
        crossover=0.4, 
        slack=1.12,
        )
    noptima, nfitness, nobjective, nfeasibility = problem.Terminate()
    for n in range(problem.nniche): 
        print(noptima[n], nfitness[n], nobjective[n], nfeasibility[n])
    PrintTimekeeper()
    print('-'*20)
    problem.Step(
        maxiter=100, 
        popsize=100, 
        elitek=0.2,
        tournk=0.8,
        tournsize=2,
        mutation=0.1, 
        sigma=1.0,
        crossover=0.4, 
        slack=1.12,
        new_niche=1,
        )
    noptima, nfitness, nobjective, nfeasibility = problem.Terminate()
    for n in range(problem.nniche): 
        print(noptima[n], nfitness[n], nobjective[n], nfeasibility[n])
    PrintTimekeeper()
