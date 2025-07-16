# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:04:52 2025

@author: hmtha
"""


import numpy as np 
from numba import njit
from datetime import datetime as dt
from collections.abc import Callable
from scipy.spatial import Delaunay
import warnings

from termination_criteria import MultiConvergence, Convergence, Maxiter, GradientStagnation

from timekeeper import timekeeper, keeptime, PrintTimekeeper
on_switch=False

#%%

# =============================================================================
    

# TODO: track metrics 
    # TODO: volume additions
        # Of optima of niches? Is this only relevant when finding extrema?
        # Maybe we can track extrema separately? But difficult in high D 
    # TODO: Saving data for a plot of convergence over time 

# TODO: reconsider infeasibility penalties??

# TODO: offload intense bits of Add_niches to njit

# TODO: update fitness so as not to calculate it for optimum niche
    # Low priority - fitness is an inexpensive calculation


# --T-O-D-O--: Restart capability
    # Actually, this should be problem specific and implemented via the `x0` kwarg

# =============================================================================

# =============================================================================
# Musing: very low population very high niches?? how does that behave
# =============================================================================

class Problem:
    @keeptime("init", on_switch)
    def __init__(
            self,
            func,
            bounds, # (lb[:], ub[:])
            integrality: np.ndarray[bool] = None,
            
            x0: Callable[..., ...]|np.ndarray|str = "uniform", 

            maximize = False,
            vectorized = False, # whether objective function accepts vectorsied array of points
            fargs = (),
            fkwargs = {},
            
            known_optimum = None, # None or np.array([coords])
            random_seed = None,
            # **kwargs,
            ):
        self._initiated = False
        
        self.rng = np.random.default_rng(random_seed)
        self.stable = random_seed != None
        if random_seed is not None and callable(x0):
            warnings.warn("""random_seed was supplied but a generator was supplied to `x0` 
                          which may be random but is not seeded. Try passing the name of 
                          a `numpy.random.Generator` distribution as a string""", UserWarning)
        
        self.func = func
        self.vectorized = vectorized
        self.fargs = fargs
        self.fkwargs = fkwargs
        self.x0 = x0

        assert isinstance(maximize, bool)
        self.maximize = maximize
        self.noptimal_obj = -np.inf if self.maximize else np.inf

        # `bounds` is a tuple containing (lb, ub)
        self.lb, self.ub = self.bounds = bounds 
        if integrality is None:
            self.mutFunction = mutGaussianFloatVec
            self.integrality = np.zeros(len(self.lb), np.bool_)
        else: 
            self.mutFunction = mutGaussianMixedVec
            self.integrality = integrality if integrality is not None else np.zeros(len(self.lb), np.bool_)
            
        assert islistlike(self.lb)
        assert islistlike(self.ub)
        assert len(self.lb) == len(self.ub)
        assert len(self.integrality) == len(self.lb)
        self.boolean_mask = (((self.ub-self.lb)==1) * (self.integrality)).astype(np.bool_)
        
        
        self.ndim = len(self.lb)
        self.cx = cxTwoPoint if self.ndim > 2 else cxOnePoint

        self.centre = (self.ub - self.lb) / 2 + self.lb
        if known_optimum is None: 
            self.optimum = self.centre.copy()
        else: 
            assert islistlike(known_optimum)
            self.optimum = np.array(known_optimum)
            assert len(self.optimum) == self.ndim
        if self.vectorized:
            self.optimal_obj = func(np.array([self.optimum]), *self.fargs)[0]
        else:
            self.optimal_obj = func(self.optimum, *self.fargs)
        
    def Initiate(
            self, 
            nniche: int, 
            popsize: int,
            ):
        
        assert nniche >= 2
        self.nniche = nniche
        self.popsize = popsize

        self.start_time = dt.now()
        self._step = 0

        ## Initiate population
        # self.population is list of lists like a 3d array of shape (nniche, popsize, ndim)        
        self.Initiate_population(self.x0)
        del self.x0
        apply_bounds_vec(self.population, self.lb, self.ub)
        
        # evaluate fitness of initial population
        self.Evaluate_objective()
        self.Update_optimum()
        self.Evaluate_feasibility()
        self.Evaluate_fitness()
        
        self._initiated = True
        
    @keeptime("Initiate_population", on_switch)
    def Initiate_population(self, x0):
        self.population = np.empty((self.nniche, self.popsize, self.ndim))
        self.objective = np.empty((self.nniche, self.popsize), dtype=np.float64)
        self.feasibility = np.empty((self.nniche, self.popsize), dtype=np.bool_)
        self.fitness = np.empty((self.nniche, self.popsize), dtype=np.float64)
        self.centroids = np.empty((self.nniche, self.ndim), np.float64)
        self.niche_elites = np.empty((self.nniche-1, self.ndim))
        
        if hasattr(x0, "__call__"):
            for i in range(self.nniche):
                for j in range(self.popsize):
                    self.population[i, j, :] = x0(self.lb, self.ub)
        if isinstance(x0, str):
            rng_dist = getattr(self.rng, x0)
            for i in range(self.nniche):
                for j in range(self.popsize):
                    self.population[i, j, :] = rng_dist(self.lb, self.ub)

        else:
            assert isinstance(x0, np.ndarray)
            assert x0.ndim == 3
            assert x0.shape[0] == self.nniche
            assert x0.shape[1] == 1 or x0.shape >= self.popsize 
            assert x0.shape[2] == self.ndim
            
            if x0.shape[1] > 1:
                njit_deepcopy(self.population, x0[:, :self.popsize, :])
            
            else: # x0.shape[1] == 1
                sigma = 0.01*(self.ub-self.lb)
                clone_parents(self.population, x0)
                self.mutFunction(self.population, sigma, 0.8, self.rng, self.integrality, self.boolean_mask)
                apply_bounds_vec(self.population, self.lb, self.ub)
                
    @keeptime("Add_niche", on_switch)
    def Add_niche(self, new_niche):
        self.population = Add_niche_to_array(self.population, new_niche)
        self.objective = Add_niche_to_array(self.objective, new_niche)
        self.fitness = Add_niche_to_array(self.fitness, new_niche)
        self.feasibility = Add_niche_to_array(self.feasibility, new_niche)
        self.centroids = np.zeros((self.nniche + new_niche, self.ndim))
        self.niche_elites = np.empty((self.nniche + new_niche - 1, self.ndim))
        
        if self.nniche < self.ndim + 1 or self.ndim == 1 or self.new_niche_heuristic is False:
            for i in range(self.nniche, self.nniche + new_niche):
                for j in range(self.popsize):
                    for k in range(self.ndim):
                        self.population[i, j, k] = self.rng.uniform(self.lb[k], self.ub[k])
            
        else: 
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
        
            best = radii.argsort(stable=self.stable)
            best = best[~np.isinf(radii)]
            
            if new_niche < 0: 
                new_niche = min(len(best), -new_niche)
            
            for i in range(new_niche):
                nn_idx = self.nniche+new_niche-1
                self.population[nn_idx, 0] = candidates[best[-(i+1)]]
                for j in range(1, self.popsize):
                    self.population[nn_idx, j] = self.population[nn_idx, 0]

            sigma = 0.1*(self.ub-self.lb)
            self.mutFunction(self.population, sigma, 0.8, self.rng, self.integrality, self.boolean_mask)
            apply_bounds_vec(self.population, self.lb, self.ub)
        
        self.Evaluate_objective(new_niche)
        self.nniche += new_niche
        self.Evaluate_fitness()
        

        return 
    
    def popsize_safe_select_parents(self, new_popsize):
        old_to_new = self.popsize/new_popsize
        elitek = int(self.elitek * old_to_new)
        tournk = int(self.tournk * old_to_new)
        tournsize = int(max(self.tournsize, min(2, old_to_new)))
        
        safe_parents = np.empty((self.nniche, tournk + elitek, self.ndim))
        
        _select_parents(
                safe_parents, self.population, self.fitness, self.objective, 
                self.feasibility, elitek, tournk, tournsize, 
                self.rng, self.maximize, self.stable)
        
        for i in range(self.nniche):
            self.rng.shuffle(safe_parents[i, :, :])
        clone_parents(self.parents, safe_parents)
        
        return
    
    @keeptime("Step", on_switch)
    def Step(
            self, 
            maxiter: int|float = np.inf, # max no of iterations in this step
            popsize: int = 100, # max no of individuals in each niche
            niche_elitism: bool = True, # clone fittest parent in each niche
            elitek: int|float = 0.1, # number of parents selected as absolute best
            tournk: int|float = 0.9, # number of parents selected via tournament
            tournsize: int = 4, # tournament size parameter
            mutation: float|tuple[float] = (0.5, 0.75), # mutation probability
            sigma: float|tuple[float] = 0.1, # standard deviation of gaussian noise for mutation (gets scaled)
            crossover: float|tuple[float] = 0.4, # crossover probability
            slack: float = np.inf, # noptimal slack in range (1.0, inf)
            new_niche: int = 0, 
            new_niche_heuristic: bool = True,
            disp_rate: int = 0,
            convergence: Callable[..., bool]|list[Callable[..., bool]] = None, 
            callback: Callable[..., ...] = None,
            ):
        if not self._initiated:
            if not new_niche > 0:
                raise Exception("""Supply `new_niche: int > 0` to initiate problem.""" )
            self.Initiate(
                nniche = new_niche, 
                popsize = popsize
                )
            new_niche=0
        
        # if new_niche is an int in (0, inf), then generate {new_niche}
        #     points - use delaunay for as many as possible and random generation if not
        # if new_niche is an int in (-inf, 0) then generate up to 
        #     {abs(new_niche)} points using delaunay exclusively 
        self.new_niche_heuristic = new_niche_heuristic
        if new_niche > 0:
            self.Add_niche(new_niche)

        ### Hyperparameters related to parent selection
        self.tournsize = tournsize
        assert elitek!=-1 or tournk !=-1, "only 1 of elitek and tournk may be -1"
        self.elitek = elitek if isinstance(elitek, int) else int(elitek*popsize)
        self.tournk = tournk if isinstance(tournk, int) else int(tournk*popsize)
        self.elitek = popsize - self.tournk if self.elitek == -1 else self.elitek
        self.tournk = popsize - self.elitek if self.tournk == -1 else self.tournk
        assert self.elitek + self.tournk <= popsize, "elitek + tournk should be weakly less than popsize"
        assert popsize > self.tournsize, "tournsize should be less than popsize"
        
        self.parents = np.empty((self.nniche, self.tournk + self.elitek, self.ndim), 
                                dtype=np.float64)
       
        if popsize == self.popsize or popsize is None:
            self.Select_parents()
        else: 
            self.popsize_safe_select_parents(popsize)
            # Change popsize here
            self.popsize = popsize
            self.population = np.empty((self.nniche, self.popsize, self.ndim), dtype=np.float64)
            self.objective = np.empty((self.nniche, self.popsize), dtype=np.float64)
            self.feasibility = np.empty((self.nniche, self.popsize), dtype=np.bool_)
            self.fitness = np.empty((self.nniche, self.popsize), dtype=np.float64)
            
        ### Hyperparameters related to parent breeding
        self.mutation = mutation
        self.sigma = sigma # adjust sigma for unnormalised bounds
        self.crossover = crossover
        self.niche_elitism = niche_elitism
        
        ### Near-optimal definitions
        self.slack = slack
        if self.maximize:
            self.noptimal_obj = self.optimal_obj * (1-(self.slack - 1))
        else: 
            self.noptimal_obj = self.optimal_obj*self.slack
            
        ### Program Control
        if convergence is None:
            self.convergence = Maxiter(maxiter)
        elif hasattr(convergence, "__iter__"):
            self.convergence = MultiConvergence(
                [Maxiter(maxiter), 
                 *convergence,
                 ]
                )
        else: 
            self.convergence = MultiConvergence(
                [Maxiter(maxiter), 
                 convergence,
                 ]
                )
        self.Callback = callback
        
        if disp_rate > 0:
            self._i = 0
            while not self.convergence(self): 
                self.Loop()
                self._i+=1
                if self.maximize:
                    best = [round(float(max(obj)),2) for obj in self.objective]
                else:
                    best = [round(float(min(obj)),2) for obj in self.objective]
                # print("\r",
                #       f"iteration {_i}. Current_best: {best}. Time: {dt.now()-self.start_time}."
                #       , end="\r")
                if self._i % disp_rate == 0:
                    print(
    f"iteration {self._i}. Current_best: {best}. Time: {dt.now()-self.start_time}.")

        else: 
            self._i = 0
            while not self.convergence(self): 
                self.Loop()
                self._i += 1
        self._step+=1
    
    @keeptime("Loop", on_switch)        
    def Loop(self):
        
        if self._i != 0:
            self.Select_parents()
        
        self.mutationInst = dither_instance(self.mutation, self.rng)
        self.crossoverInst = dither_instance(self.crossover, self.rng)
        self.sigmaInst = dither_instance(self.sigma, self.rng) * (self.ub - self.lb)
        
        self.Generate_offspring()
        self.Evaluate_objective()
        self.Update_optimum()
        self.Evaluate_feasibility()
        self.Evaluate_fitness()
        if self.Callback is not None:
            self.Callback(self)
        
        
    @keeptime("Terminate", on_switch)
    def Terminate(self):
        self.Run_statistics()
        self.Return_noptima()
        return self.noptima, self.nfitness, self.nobjective, self.nfeasibility

    @keeptime("Update_optimum", on_switch)
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
                    self.optimum = self.population[n, self.objective[n].argmax(), :].copy()
        else: 
            if self.objective.min() < self.optimal_obj:
                for n in range(self.nniche):
                    if self.objective[n].min() > self.optimal_obj:
                        continue
                    self.optimal_obj = self.objective[n].min()
                    self.optimum = self.population[n, self.objective[n].argmin(), :].copy()

    @keeptime("Return_noptima", on_switch)
    def Return_noptima(self):
        if self.maximize:
            nindex = [self.objective[0].argmax()]
        else: 
            nindex = [self.objective[0].argmin()]
            
        nindex += list(best_nindex(self.fitness, self.feasibility))
            
        self.noptima = [self.population[0, nindex[0]]]
        self.noptima += [self.population[n, nindex[n]] for n in range(1, self.nniche)]
        self.nfitness = [self.fitness[n, nindex[n]] for n in range(self.nniche)]
        self.nobjective = [self.objective[n, nindex[n]] for n in range(self.nniche)]
        self.nfeasibility = [self.feasibility[n, nindex[n]] for n in range(self.nniche)]
        
    
    def Run_statistics(self):
        """TODO"""
        pass

    def Select_parents(self):
        _select_parents(
                self.parents, self.population, self.fitness, self.objective, 
                self.feasibility, self.elitek, self.tournk, self.tournsize, 
                self.rng, self.maximize, self.stable)
    
    @keeptime("Generate_offspring", on_switch)
    def Generate_offspring(self):
        clone_parents(self.population, self.parents)
        
        if self.niche_elitism:
            # We do not wan a "hall of fame" or such since fitness is not independent 
            self.niche_elites = populate_niche_elites(
                self.niche_elites, self.population, self.fitness, self.feasibility, self.objective, self.maximize)
             
        cx_vec(self.population, self.crossoverInst, self.cx, self.rng)
        self.mutFunction(self.population, self.sigmaInst, self.mutationInst, self.rng, self.integrality, self.boolean_mask)
        apply_bounds_vec(self.population, self.lb, self.ub)
        
        self.population[0, 0, :] = self.optimum
        
        if self.niche_elitism:
            for i in range (1, self.nniche):
                self.population[i, 0, :] = self.niche_elites[i-1, :]

    def Evaluate_fitness(self):
        """
        Calculates the fitness of each individual as the minimum (per-variable)
        distance from the centroids of all the other niches besides the one 
        where the individual itself sits, and including the 'optimal' niche. 
        """
        find_centroids(self.centroids, self.population)
        _evaluate_fitness(
            self.fitness, self.population, self.centroids, self.objective, self.feasibility)

    @keeptime("Evaluate_objective", on_switch)        
    def Evaluate_objective(self, new_niche=None):
        """
        Calculates the objective function for each individual
        """
        if new_niche is None:
            if self.vectorized:
                for n in range(self.nniche):
                    self.objective[n] = self.func(self.population[n], *self.fargs, **self.fkwargs)
            else: 
                for n in range(self.nniche):
                    for i in range(self.popsize):
                        self.objective[n, i] = self.func(self.population[n, i], *self.fargs, **self.fkwargs)
        else: 
            if self.vectorized:
                for n in range(self.nniche, self.nniche + new_niche):
                    self.objective[n] = self.func(self.population[n], *self.fargs, **self.fkwargs)
            else: 
                for n in range(self.nniche, self.nniche + new_niche):
                    for i in range(self.popsize):
                        self.objective[n, i] = self.func(self.population[n, i], *self.fargs, **self.fkwargs)

    def Evaluate_feasibility(self):
        _evaluate_feasibility(
            self.feasibility, self.population, self.objective, self.noptimal_obj, self.maximize)
   
#%%
    
def islistlike(obj):
    if not hasattr(obj, "__iter__"):
        return False
    if not hasattr(obj, "__len__"):
        return False
    return True

def isfunclike(obj):
    # Add other conditions?
    return callable(obj)

@njit
def Add_niche_to_array(old_array, new_niche):
    new_array = np.empty(
        (old_array.shape[0] + new_niche, 
         old_array.shape[1], 
         old_array.shape[2],
         ), 
        dtype=old_array.dtype)
    for i in range(old_array.shape[0]):
        for j in range(old_array.shape[1]):
            for k in range(old_array.shape[2]):
                new_array[i, j, k] = old_array[i, j, k]
    return new_array

@njit 
def best_nindex(fitness, feasibility):
    # first niche reserved for optimisation
    indices = np.zeros(fitness.shape[0] - 1, np.int64)
    for i in range(len(indices)):
        i_o = i + 1 
        _best = -np.inf
        if feasibility[i_o].any():
            for j in range(fitness.shape[1]):
                if feasibility[i_o, j] is True: 
                    if fitness[i_o, j] > _best:
                        _best = fitness[i_o, j]
                        indices[i] = j
        else: 
            for j in range(fitness.shape[1]):
                if fitness[i_o, j] > _best:
                    _best = fitness[i_o, j]
                    indices[i] = j
    return indices

@njit 
def njit_deepcopy(new, old):
    flat_new = new.ravel()
    flat_old = old.ravel()
    
    for i in range(flat_old.shape[0]):
        flat_new[i] = flat_old[i]
    
@keeptime("Evaluate_fitness", on_switch)        
@njit
def _evaluate_fitness(fitness, population, centroids, objective, feasibility):
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

@keeptime("Select_parents", on_switch)
@njit
def _select_parents(
        parents, population, fitness, objective, feasibility, elitek, tournk, tournsize, rng, maximize, stable):
    
    parents[0, :elitek, :] = selBest(population[0, :, :], objective[0, :], elitek, maximize, stable)
    parents[0, elitek:, :] = selTournament(
        population[0, :, :], objective[0, :], tournk, tournsize, rng, maximize)
        
    for i in range(1, parents.shape[0]):
        parents[i, :elitek, :] = selBest_fallback(
            population[i, :, :], fitness[i, :], feasibility[i, :], objective[i, :], elitek, maximize, stable)
        parents[i, elitek:, :] = selTournament_fallback(
            population[i, :, :], fitness[i, :], feasibility[i, :], objective[i, :], tournk, tournsize, rng, maximize)

@keeptime("Evaluate_feasibility", on_switch)        
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
def njit_cdist(A, B):
    """
    Numba-compatible implementation of Euclidean distance calculation,
    similar to the output of scipy.cdist(A, B, 'euclidean').
    Args:
        A (np.array): An (M, K) array of M vectors.
        B (np.array): An (N, K) array of N vectors.
    Returns:
        np.array: An (M, N) matrix of Euclidean distances.
    """
    M, K = A.shape
    N, K = B.shape
    # Calculate the squared L2 norm for each row in A and B
    A_sq = np.sum(A**2, axis=1).reshape(M, 1)
    B_sq = np.sum(B**2, axis=1).reshape(1, N)
    # Calculate the dot product term: 2 * (A @ B.T)
    dot_product = 2 * A @ B.T
    # Calculate squared Euclidean distances using the identity
    dist_sq = A_sq - dot_product + B_sq
    # Handle potential floating-point inaccuracies that result in small negative values
    dist_sq[dist_sq < 0] = 0
    return np.sqrt(dist_sq)
    
@njit
def euclideanDistance(p1, p2):
    """Euclidean distance"""
    return sum((p1-p2)**2)**0.5
    
@njit
def pointwiseDistance(p1, p2):
    return min(np.abs(p1-p2))
    
@njit
def mutFloat(item, sigma, rng):
    """ gaussian mutation for single variable """
    return rng.normal(item, sigma)

@njit
def mutInt(item, sigma, rng):
    """Integer mutation for single variable"""
    return round(rng.normal(item, sigma))

@njit
def mutBool(item,):
    """Boolean mutation for single variable"""
    return 1.0 - item

@njit 
def mutGaussianMixedVec(population, sigma, indpb, rng, integrality, boolean_mask, ignore_first=True):
    start = 1 if ignore_first is True else 0
    for i in range(population.shape[0]):
        for j in range(start, population.shape[1]):
            for k in range(population.shape[2]):
                if rng.random() < indpb:
                    if boolean_mask[k] is True:
                        population[i, j, k] = mutBool(population[i, j, k]) 
                    elif integrality[k] is True:
                        population[i, j, k] = mutInt(population[i, j, k], sigma[k], rng)
                    else: 
                        population[i, j, k] = mutFloat(population[i, j, k], sigma[k], rng)

@njit
def mutGaussianFloatVec(population, sigma, indpb, rng, integrality, boolean_mask, ignore_first=True):
    start = 1 if ignore_first is True else 0
    for i in range(population.shape[0]):
        for j in range(start, population.shape[1]):
            for k in range(population.shape[2]):
                if rng.random() < indpb:
                    population[i, j, k] = mutFloat(population[i, j, k], sigma[k], rng)

@njit
def cx_vec(population, crossoverInst, cx, rng):
    for i in range(population.shape[0]):
        # shuffle parents for mating
        rng.shuffle(population[i])
        
        for ind1, ind2 in zip(population[i, ::2], population[i, 1::2]):
            # crossover probability
            if rng.random() < crossoverInst:
                # Apply crossover
                cx(ind1, ind2, rng)

@njit
def _do_cx(ind1, ind2, index1, index2):
    buffer = 0.0
    # modify in place
    for i in range(index1, index2):
        buffer = ind1[i]
        ind1[i] = ind2[i]
        ind2[i] = buffer   

@njit
def cxOnePoint(ind1, ind2, rng):
    index1 = rng.integers(0, len(ind1))
    _do_cx(ind1, ind2, index1, len(ind1))
    
@njit
def cxTwoPoint(ind1, ind2, rng):
    # +1 / -1 adjustments are made to ensure there is always a crossover 
    # only valid for ndim >= 3
    index1 = rng.integers(0, len(ind1)-1)
    index2 = rng.integers(index1+1, len(ind1))
    _do_cx(ind1, ind2, index1, index2)

@keeptime("_selT_draw_indices", on_switch)
@njit 
def _selT_draw_indices(indices, ub, rng):
    for i in range(indices.size):
        indices[i] = rng.integers(0, ub)

@keeptime("selTournament_fallback", on_switch)
@njit
def selTournament_fallback(niche, fitness, feasibility, objective, n, tournsize, rng, maximize):
    """select on fitness preferred. fitness always maximised. 
    objective max/minimized based on value of `maximize`"""
    selected = np.empty((n, niche.shape[1]), np.float64)
    indices = np.empty(tournsize, np.int64)
    feasibility_threshold = tournsize / 2 
    
    for m in range(n):
        _selT_draw_indices(indices, niche.shape[0], rng)
        
        _feas = 0
        for idx in indices:
            if feasibility[idx]:
                _feas+=1 
        
        if _feas <= feasibility_threshold: # mostly infeasible
            _selected_idx = _do_selTournament(objective, maximize, indices)
        else: # mostly feasible
            _selected_idx = _do_selTournament(fitness, True, indices)

        selected[m, :] = niche[_selected_idx, :]
    
    return selected 

@keeptime("selTournament", on_switch)
@njit
def selTournament(niche, objective, n, tournsize, rng, maximize):
    """objective max/minimized based on value of `maximize`"""
    selected = np.empty((n, niche.shape[1]), np.float64)
    indices = np.empty(tournsize, np.int64)
    
    for m in range(n):
        _selT_draw_indices(indices, niche.shape[0], rng)
        _selected_idx = _do_selTournament(objective, maximize, indices)
        selected[m, :] = niche[_selected_idx, :]

    return selected # , selected_value

@njit 
def _do_selTournament(criteria, maximize, indices):
    if maximize:
        _selected_idx = 0
        _best = -np.inf
        for idx in indices:
            if criteria[idx] > _best:
                _selected_idx = idx
                _best = criteria[idx]
    else:
        _selected_idx = 0
        _best = np.inf
        for idx in indices:
            if criteria[idx] < _best:
                _selected_idx = idx
                _best = criteria[idx]
    return _selected_idx 

@njit
def _stabilise_sort(indices, values):
    """Sorts blocks of duplicate values within a list of indices
    in-place based on the index values for a stable order."""
    i = 1
    while i < len(indices):
        # Check if the value at the current index is the same as the previous one
        if values[indices[i]] == values[indices[i-1]]:
            # Found the start of a block of equal-valued items
            start_block = i - 1
            # Find the end of the block
            while i < len(indices) and values[indices[i]] == values[indices[start_block]]:
                i += 1
            end_block = i
            # Sort the slice of indices corresponding to the duplicates.
            # This provides a stable, deterministic order.
            indices[start_block:end_block].sort()
        else:
            i += 1
    return indices

@keeptime("selBest_fallback", on_switch)
@njit
def selBest_fallback(niche, fitness, feasibility, objective, n, maximize, stable):
    """ Selects best `n` individuals based on fitness.
    If there are not `n` feasible individuals, selects on objective"""
    
    _feas = 0 
    for i in range(len(feasibility)):
        if feasibility[i]:
            _feas += 1 
    
# =============================================================================
### Alteranative code block. Much slower 
#     if _feas < n: # mostly infeasible
#         return selBest(niche, objective, n, maximize, stable)
#     
#     selected = np.empty((n, niche.shape[1]))
#     indices = np.empty(n, np.int64)
#     
#     feasible_indices = np.where(feasibility)[0]
#     feasible_fitness = fitness[feasible_indices]
#     
#     if _feas == n: # edge case breaks numba but also we can skip and be more efficient anyway
#         indices[:] = feasible_indices
#     elif stable: 
# =============================================================================

# =============================================================================
#   This code block makes the entire algorithm better than 10x faster 
#   Requires a slightly dodgy alteration of program logic with _feas < / <= n
#   But I think this simplification is insignificant 
    if _feas <= n: # mostly infeasible
        return selBest(niche, objective, n, maximize, stable)
    
    selected = np.empty((n, niche.shape[1]))
    indices = np.empty(n, np.int64)
    
    feasible_indices = np.where(feasibility)[0]
    feasible_fitness = fitness[feasible_indices]
    
    if stable: 
# =============================================================================
        _indices = np.argsort(feasible_fitness)
        _stabilise_sort(_indices, feasible_fitness)
        indices[:] = feasible_indices[_indices[-n:]]
    else: 
        indices[:] = feasible_indices[np.argpartition(feasible_fitness, -n)[-n:]]
    
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
        
    return selected 

@keeptime("selBest", on_switch)
@njit
def selBest(niche, objective, n, maximize, stable):
    selected = np.empty((n, niche.shape[1]))
    indices = np.empty(n, np.int64)
    
    if stable:
        _indices = np.argsort(objective)
        _stabilise_sort(_indices, objective)
        if maximize: 
            indices[:] = _indices[-n:]
        else: 
            indices[:] = _indices[:n]
    else: 
        # This is much faster but does not preserve order 
        if maximize:
            indices = np.argpartition(objective, -n)[-n:]
        else:
            indices = np.argpartition(objective, n)[:n]   
        
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
    
    return selected 


@keeptime("selBest1", on_switch)
@njit
def selBest1(niche, objective, maximize):
    """Special case of selBest where n=1"""
    if maximize:
        index = objective.argmax()
    else:
        index = objective.argmin()
    return niche[index, :] 

@keeptime("selBest_fallback", on_switch)
@njit
def selBest1_fallback(niche, fitness, feasibility, objective, maximize):
    """ Special case of selBest_fallback where n = 1
    Selects best `n` individuals based on fitness.
    If there are not `n` feasible individuals, selects on objective"""
    
    _feas = False
    for i in range(len(feasibility)):
        if feasibility[i]:
            _feas = True 
            break
    
    if not _feas:
        return selBest1(niche, objective, maximize)
    else: 
        index = fitness.argmax()
        
    return niche[index, :] 

@njit
def populate_niche_elites(niche_elites, population, fitness, feasibility, objective, maximize):
    for i in range(1, 1 + niche_elites.shape[0]):
        niche_elites[i-1, :] = selBest1_fallback(
            population[i], fitness[i], feasibility[i], objective[i], maximize)
    return niche_elites

@keeptime("find_centroids", on_switch)
@njit
def find_centroids(centroids, population):
    centroids[:,:] = 0.0
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                centroids[i, k] += population[i, j, k]
    centroids /= population.shape[1]

def dither_instance(parameter, rng):
    if hasattr(parameter, "__iter__"):
        return _dither(*parameter, rng)
    else:
        return parameter

@njit
def _dither(lower, upper, rng):
    return rng.uniform(lower, upper)

@keeptime("apply_bounds", on_switch)
@njit
def apply_bounds(ind, lb, ub):
    for i in range(len(ind)):
        ind[i] = max(lb[i], ind[i])
        ind[i] = min(ub[i], ind[i])
    return ind

@njit
def apply_bounds_vec(population, lb, ub):
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                population[i, j, k] = min(ub[k], max(lb[k], population[i, j, k]))
                
@njit
def clone_parents(population, parents):
    nparents = parents.shape[1]
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            jn = j%nparents
            for k in range(population.shape[2]):
                population[i, j, k] = parents[i, jn, k]
    


#%% 


    
if __name__ == "__main__":


    @njit
    def Objective(values_array): 
        """ Vectorized = True """
        # For the first test, ndim=2 and works for a function with two decision variables
        
        z = 2 + np.zeros(values_array.shape[0], np.float64)
        
        # Perform summation with a guaranteed order
        for i in range(values_array.shape[0]):
            for j in range(values_array.shape[1]):
                z[i] += np.sin(19 * np.pi * values_array[i, j]) + values_array[i, j] / 1.7
        return z        
        
    # @njit
    # def Objective(values_array): 
    #     """ Vectorized = True """
    #     # For the first test, ndim=2 and works for a function with two decision variables
    #     z = np.sum(np.sin(19 * np.pi * values_array) + values_array / 1.7, axis=1) + 2
    #     return z
    
    # def Objective(x):
    #     return ((0.5*(x-0.55)**2 + 0.9)*np.sin(5*np.pi*x)**6)[0]
    
    lb = 0*np.ones(2)
    ub = 1*np.ones(2)
    
    # alternatives = np.array([[0.021, 0.974],
    #                [0.974, 0.021],
    #                [0.443, 0.549]])
    # x0 = np.dstack(tuple(np.repeat(alternatives[n], 100).reshape(2, 100).T for n in range(3))).transpose(2, 0, 1)
    
    # x0 = np.array([[[0.97374456, 0.97345396]], 
    #                 [[0.02255059, 0.97683429]],
    #                 [[0.97812274, 0.02309379]], 
    #                 # [[0.76559229, 0.34602654]],
    #                 ])
    
    problem = Problem(
        func=Objective,
        bounds = (lb, ub),
        maximize = True, 
        vectorized = True,
        # random_seed = 1,
        # x0 = x0,
        )

    # problem.Step(
    #     maxiter=400, 
    #     popsize=800, 
    #     elitek=0.5,
    #     tournk=0.5,
    #     tournsize=3,
    #     mutation=0.5, 
    #     sigma=(0.01, 0.4),
    #     crossover=0.0, 
    #     slack=1.12,
    #     disp_rate=25,
    #     niche_elitism = True, 
    #     new_niche = 3,
    #     )
    # noptima, nfitness, nobjective, nfeasibility = problem.Terminate()
    # for n in range(problem.nniche): 
    #     print(noptima[n], nfitness[n], nobjective[n], nfeasibility[n])

    NICHE_ELITISM = True

    problem.Step(
        maxiter=10, 
        popsize=100, 
        elitek=0.25,
        tournk=-1,
        tournsize=2,
        mutation=0.5, 
        sigma=0.4,
        crossover=0.3, 
        slack=1.12,
        disp_rate=25,
        niche_elitism = NICHE_ELITISM, 
        new_niche = 3,
        )
    noptima, nfitness, nobjective, nfeasibility = problem.Terminate()
    for n in range(problem.nniche): 
        print(noptima[n], nfitness[n], nobjective[n], nfeasibility[n])
        
        
    # PrintTimekeeper(on_switch=on_switch)
    print('-'*20)
    problem.Step(
        maxiter=200, 
        popsize=300, 
        elitek=0.2,
        tournk=-1,
        tournsize=2,
        mutation=0.4, 
        sigma=0.15,
        crossover=0.3, 
        slack=1.12,
        # new_niche=1,
        disp_rate=20,
        niche_elitism = NICHE_ELITISM, 
        )
    noptima, nfitness, nobjective, nfeasibility = problem.Terminate()
    for n in range(problem.nniche): 
        print(noptima[n], nfitness[n], nobjective[n], nfeasibility[n])
    # PrintTimekeeper(on_switch=on_switch)
    print('-'*20)

    problem.Step(
        maxiter=10, 
        popsize=1000, 
        elitek=0.4,
        tournk=-1,
        tournsize=2,
        mutation=0.5, 
        sigma=0.005,
        crossover=0.3, 
        slack=1.12,
        # new_niche=1,
        disp_rate=5,
        niche_elitism = NICHE_ELITISM, 
        )
    noptima, nfitness, nobjective, nfeasibility = problem.Terminate()
    for n in range(problem.nniche): 
        print(noptima[n], nfitness[n], nobjective[n], nfeasibility[n])
    # PrintTimekeeper(on_switch=on_switch)

        
    PrintTimekeeper(on_switch=on_switch)
