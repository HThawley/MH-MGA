# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:04:52 2025

@author: hmtha
"""


import numpy as np 
from numba import njit
from datetime import datetime as dt

from scipy.spatial import Delaunay

from termination_criteria import MultiCriteriaConvergence, ConvergenceCriterion, Maxiter, GradientStagnation

from timekeeper import timekeeper, keeptime, PrintTimekeeper
on_switch=False

#%%

# =============================================================================
# TODO: update selection tournaments (fallback)
    # TODO: seltournament should select on fitness/objective on a draw-by-draw 
        # basis not a niche by niche basis 
    # TODO: selBest should always prioritise feasible solutions
    
# TODO: Consider staging noptimal slack lowering
    # for population stability when "annealing" (lowering noptimal slack)
    # progressively reduce it to maintain population stability?

# TODO: track metrics 
    # TODO: stagnation criteria for optimisation 
    # TODO: volume additions
        # Of optima of niches? Is this only relevant when finding extrema?
        # Maybe we can track extrema separately? But difficult in high D 
    # TODO: Saving data for a plot of convergence over time 

# TODO: when returning noptima, return the most most fit that is feasible

# TODO: reconsider infeasibility penalties??

# TODO: hyperparameter tuning skeleton
    # To be done after track metrics as metrics needed for termination
    
# TODO: offload intense bits of Add_niches to njit
# TODO: offload intense bits of Generate_offspring to njit

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
            bounds, # [lb[:], ub[:]]
            
            nniche: int, 
            popsize: int,
            x0 = np.random.uniform, # callable[...]|np.ndarray
            maximize = False,
            vectorized = False, # whether objective function accepts vectorsied array of points
            fargs = (),
            
            known_optimum = None, # None or np.array([coords])
                    
            # **kwargs,
            ):
        
        self.func = func
        self.vectorized = vectorized
        self.fargs = fargs

        assert isinstance(maximize, bool)
        self.maximize = maximize
        
        # `bounds` is a tuple containing (lb, ub)
        self.lb, self.ub = self.bounds = bounds 
        assert islistlike(self.lb)
        assert islistlike(self.ub)
        assert len(self.lb) == len(self.ub)
        self.ndim = len(self.lb)

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
        
        assert nniche >= 2
        self.nniche = nniche
        self.popsize = popsize
        self.noptimal_obj = -np.inf if self.maximize else np.inf

        self.start_time = dt.now()

        ## Initiate population
        # self.population is list of lists like a 3d array of shape (nniche, popsize, ndim)        
        self.Initiate_population(x0)
        apply_bounds_vec(self.population, self.lb, self.ub)
        
        # evaluate fitness of initial population
        self.Evaluate_objective()
        self.Update_optimum()
        self.Evaluate_feasibility()
        self.Evaluate_fitness()
    
    @keeptime("Instantiate_working_arrays", on_switch)
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

    @keeptime("Initiate_population", on_switch)
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
            assert x0.ndim == 3
            assert x0.shape[0] == self.nniche
            assert x0.shape[1] == 1 or x0.shape >= self.popsize 
            assert x0.shape[2] == self.ndim
            
            if x0.shape[1] > 1:
                njit_deepcopy(self.population, x0[:, :self.popsize, :])
            
            else: # x0.shape[1] == 1
                sigma = 0.01*(self.ub-self.lb)
                clone_parents(self.population, x0)
                mutGaussian_vec(self.population, sigma, 0.8, True)
                apply_bounds_vec(self.population, self.lb, self.ub)
                
    @keeptime("Add_niche", on_switch)
    def Add_niche(self, new_niche):
        old_pop = self.population.copy()
        self.population = np.empty((self.nniche + new_niche, self.popsize, self.ndim))
        for i in range(self.nniche):
            for j in range(self.popsize):
                for k in range(self.ndim):
                    self.population[i, j, k] = old_pop[i, j, k]
        
        if self.nniche < self.ndim + 1 or self.ndim == 1 or self.new_niche_heuristic is False:
            for i in range(self.nniche, self.nniche + new_niche):
                for j in range(self.popsize):
                    for k in range(self.ndim):
                        self.population[i, j, k] = np.random.uniform(self.lb[k], self.ub[k])
            
            self.nniche += new_niche
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
        best = best[~np.isinf(radii)]
        
        if new_niche < 0: 
            new_niche = min(len(best), -new_niche)
        
        sigma = 0.1*(self.ub-self.lb)
        for i in range(new_niche):
            nn_idx = self.nniche+new_niche-1
            self.population[nn_idx, 0] = candidates[best[-(i+1)]]
            for j in range(1, self.popsize):
                self.population[nn_idx, j] = self.population[nn_idx, 0]
                
        mutGaussian_vec(self.population, sigma, 0.8, True)        
        apply_bounds_vec(self.population, self.lb, self.ub)
        self.nniche += new_niche
        return 
    
    @keeptime("Step", on_switch)
    def Step(
            self, 
            maxiter: int|float = np.inf, # max no of iterations in this step
            popsize: int = 100, # max no of individuals in each niche
            elitek: int|float = 0.1, # number of parents selected as absolute best
            tournk: int|float = 0.9, # number of parents selected via tournament
            tournsize: int = 4, # tournament size parameter
            mutation: float|tuple[float] = (0.5, 1.5), # mutation probability
            sigma: float|tuple[float] = 0.1, # standard deviation of gaussian noise for mutation (gets scaled)
            crossover: float|tuple[float] = 0.4, # crossover probability
            slack: float = np.inf, # noptimal slack in range (1.0, inf)
            new_niche: int = 0, 
            new_niche_heuristic: bool = True,
            disp_rate: int = 0,
            convergence = None, #: callable|list[callable] 
            callback: callable = None,
            ):
        # if new_niche is an int in (0, inf), then generate {new_niche}
        #     points - use delaunay for as many as possible and random generation if not
        # if new_niche is an int in (-inf, 0) then generate up to 
        #     {abs(new_niche)} points using delaunay exclusively 
        
        self.new_niche_heuristic = new_niche_heuristic
        if new_niche > 0:
            self.Add_niche(new_niche)

        self.popsize = popsize
        self.callback = callback
        
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
        
        assert self.elitek + self.tournk <= popsize, "elitek + tournk should be weakly less than popsize"
        assert self.popsize > self.tournsize, "tournsize should be less than popsize"
        
        if self.ndim > 2: 
            self.cx = cxTwoPoint
        else: 
            self.cx = cxOnePoint
            
        if convergence is None:
            self.convergence = Maxiter(maxiter)
        elif hasattr(convergence, "__iter"):
            self.convergence = MultiCriteriaConvergence(
                [Maxiter(maxiter), 
                 *convergence,
                 ]
                )
        else: 
            self.convergence = MultiCriteriaConvergence(
                [Maxiter(maxiter), 
                 convergence,
                 ]
                )
        
        self.Instantiate_working_arrays()
        if disp_rate > 0:
            _i=0
            while not self.convergence(
                    value=self.optimal_obj,
                    ): 
                self.Loop()
                _i+=1
                if self.maximize:
                    best = [round(float(max(obj)),2) for obj in self.objective]
                else:
                    best = [round(float(min(obj)),2) for obj in self.objective]
                # print("\r",
                #       f"iteration {_i}. Current_best: {best}. Time: {dt.now()-self.start_time}."
                #       , end="\r")
                if _i % disp_rate == 0:
                    print(
    f"iteration {_i}. Current_best: {best}. Time: {dt.now()-self.start_time}.")

        else: 
            while not self.convergence(
                    value=self.optimal_obj,
                    ): 
                self.Loop()
    
    
    @keeptime("Loop", on_switch)        
    def Loop(self):
        
        self.mutationInst = dither_instance(self.mutation)
        self.crossoverInst = dither_instance(self.crossover)
        self.sigmaInst = dither_instance(self.sigma) * (self.ub - self.lb)
        
        self.Select_parents()
        self.Generate_offspring()
        self.Evaluate_objective()
        self.Update_optimum()
        self.Evaluate_feasibility()
        self.Evaluate_fitness()
        
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
                    self.optimum = self.population[n, self.objective[n].argmax(), :]
        else: 
            if self.objective.min() < self.optimal_obj:
                for n in range(self.nniche):
                    if self.objective[n].min() > self.optimal_obj:
                        continue
                    self.optimal_obj = self.objective[n].min()
                    self.optimum = self.population[n, self.objective[n].argmin(), :]

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
                self.feasibility, self.elitek, self.tournk, self.tournsize, self.maximize)
    
    @keeptime("Generate_offspring", on_switch)
    def Generate_offspring(self):
        self.population = np.empty((self.nniche, self.popsize, self.ndim))
        clone_parents(self.population, self.parents)
        
        for niche in self.population:
            # shuffle parents for mating
            np.random.shuffle(niche)
            
            for ind1, ind2 in zip(niche[::2], niche[1::2]):
                # crossover probability
                if np.random.random() < self.crossoverInst:
                    # Apply crossover
                    self.cx(ind1, ind2)
        mutGaussian_vec(self.population, self.sigmaInst, self.mutationInst)
        apply_bounds_vec(self.population, self.lb, self.ub)

        self.population[0][0] = self.optimum

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
        # =============================================================================
        if self.maximize: 
            _evaluate_fitness_max(
                self.fitness, self.population, self.centroids, self.objective, self.feasibility)
        else:
            _evaluate_fitness_min(
                self.fitness, self.population, self.centroids, self.objective, self.feasibility)

    @keeptime("Evaluate_objective", on_switch)        
    def Evaluate_objective(self):
        """
        Calculates the objective function for each individual
        """
        if self.vectorized:
            for n in range(self.nniche):
                self.objective[n] = self.func(self.population[n], *self.fargs)
        else: 
            for n in range(self.nniche):
                for i in range(self.popsize):
                    self.objective[n, i] = self.func(self.population[n, i], *self.fargs)

    def Evaluate_feasibility(self):
        _evaluate_feasibility(
            self.feasibility, self.population, self.objective, self.noptimal_obj, self.maximize)
   
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
    
@keeptime("Evaluate_fitness_max", on_switch)        
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
                fitness[i, j] -= (abs(objective[i, j])*1000 + 1000)

@keeptime("Evaluate_fitness_min", on_switch)        
@njit
def _evaluate_fitness_min(fitness, population, centroids, objective, feasibility):
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
                fitness[i, j] += (abs(objective[i, j])*1000 + 1000)

@keeptime("Select_parents", on_switch)
@njit
def _select_parents(
        parents, population, fitness, objective, feasibility, elitek, tournk, tournsize, maximize):
    
    dummy = selBest(population[0, :, :], objective[0, :], elitek, maximize)
    for j in range(dummy.shape[0]):
        for k in range(dummy.shape[1]):
            parents[0, j, k] = dummy[j, k]
    dummy = selTournament(
        population[0, :, :], objective[0, :], tournk, tournsize, maximize)
    for j, jp in enumerate(range(elitek, elitek+dummy.shape[0])):
        for k in range(dummy.shape[1]):
            parents[0, jp, k] = dummy[j, k]
    # parents[0, :elitek, :] = selBest(population[0, :, :], objective[0, :], elitek, maximize)
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
            ind[i] = np.random.normal(ind[i], sigma[i])
    return ind

@njit
def mutGaussian_vec(population, sigma, indpb, ignore_first=True):
    if ignore_first is True:
        start=1
    else: 
        start=0
    for i in range(population.shape[0]):
        for j in range(start, population.shape[1]):
            for k in range(population.shape[2]):
                if np.random.random() < indpb:
                    population[i, j, k] = np.random.normal(population[i, j, k], sigma[k])

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

@keeptime("_selT_draw_indices", on_switch)
@njit 
def _selT_draw_indices(indices, n, size):
    indices[:] = np.random.randint(0, n, size=size)

@keeptime("selTournament_fallback", on_switch)
@njit
def selTournament_fallback(niche, fitness, feasibility, objective, n, tournsize, maximize):
    """select on fitness preferred. fitness always maximised. 
    objective max/minimized based on value of `maximize`"""
    selected = np.empty((n, niche.shape[1]), np.float64)
    # selected_value = np.empty(n, np.float64)
    indices = np.empty(tournsize, np.int64)
    feasibility_threshold = tournsize / 2 
    
    for m in range(n):
        _selT_draw_indices(indices, niche.shape[0], tournsize)
        
        _feas = 0
        for idx in indices:
            if feasibility[idx]:
                _feas+=1 
        
        if _feas <= feasibility_threshold: # mostly infeasible
            if maximize:
                _selected_idx = 0
                _best = -np.inf
                for idx in indices:
                    if objective[idx] > _best:
                        _selected_idx = idx
                        _best = objective[idx]
            else: 
                _selected_idx = 0
                _best = np.inf
                for idx in indices:
                    if objective[idx] < _best:
                        _selected_idx = idx
                        _best = objective[idx]
        
        else: # mostly feasible
            _selected_idx = 0
            _best = -np.inf
            for idx in indices:
                if objective[idx] > _best:
                    _selected_idx = idx
                    _best = objective[idx]
                    
        # selected_value[m] = _best
        selected[m, :] = niche[_selected_idx, :]
    
    return selected # , selected_value

@keeptime("selTournament", on_switch)
@njit
def selTournament(niche, objective, n, tournsize, maximize):
    """objective max/minimized based on value of `maximize`"""
    selected = np.empty((n, niche.shape[1]), np.float64)
    # selected_value = np.empty(n, np.float64)
    indices = np.empty(tournsize, np.int64)
    if maximize:
        for m in range(n):            
            _selT_draw_indices(indices, niche.shape[0], tournsize)
            _selected_idx = 0
            _best = -np.inf
            for idx in indices:
                if objective[idx] > _best:
                    _selected_idx = idx
                    _best = objective[idx]
            # selected_value[m] = _best
            selected[m, :] = niche[_selected_idx, :]
    else:
        for m in range(n):            
            _selT_draw_indices(indices, niche.shape[0], tournsize)
            _selected_idx = 0
            _best = np.inf
            for idx in indices:
                if objective[idx] < _best:
                    _selected_idx = idx
                    _best = objective[idx]
            # selected_value[m] = _best
            selected[m, :] = niche[_selected_idx, :]

    return selected # , selected_value

@keeptime("selBest_fallback", on_switch)
@njit
def selBest_fallback(niche, fitness, feasibility, objective, n, maximize):
    """ Selects best `n` individuals based on fitness.
    If there are not `n` feasible individuals, selects on objective"""
    
    selected = np.empty((n, niche.shape[1]))
    # selected_value = np.empty(n)
    indices = np.empty(n, np.int64)
    
    _feas = 0 
    for i in range(len(feasibility)):
        if feasibility[i]:
            _feas += 1 
    
    if _feas < n: # mostly infeasible
        if maximize:
            indices = np.argsort(objective)[-n:]
        else: 
            indices = np.argsort(objective)[:n]
        # selected_value = objective[indices]
    else: 
        indices = np.argsort(fitness)[-n:]
        # selected_value = fitness[indices]
        
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
        
    return selected # , selected_value  

@keeptime("selBest", on_switch)
@njit
def selBest(niche, objective, n, maximize):
    selected = np.empty((n, niche.shape[1]))
    indices = np.empty(n, np.int64)
    # selected_value = np.empty(n)
    if maximize:
        indices = np.argsort(objective)[-n:]
    else:
        indices = np.argsort(objective)[:n]   
        
    for j in range(n):
        selected[j, :] = niche[indices[j], :]
        # selected_value[j] = objective[indices[j]]
    
    return selected # , selected_value

@keeptime("find_centroids", on_switch)
@njit
def find_centroids(centroids, population):
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                centroids[i, k] += population[i, j, k]
    centroids /= population.shape[1]

def dither_instance(parameter):
    if hasattr(parameter, "__iter__"):
        return _dither(*parameter)
    else:
        return parameter

@njit
def _dither(lower, upper, distribution=np.random.uniform):
    return distribution(lower, upper)

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
    
    x0 = np.array([[[0.97374456, 0.97345396]], 
                    [[0.02255059, 0.97683429]],
                    [[0.97812274, 0.02309379]], 
                    # [[0.76559229, 0.34602654]],
                    ])
    
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
        mutation=0.3, 
        sigma=0.5,
        crossover=0.4, 
        slack=1.12,
        disp_rate=25,
        )
    noptima, nfitness, nobjective, nfeasibility = problem.Terminate()
    for n in range(problem.nniche): 
        print(noptima[n], nfitness[n], nobjective[n], nfeasibility[n])
    # PrintTimekeeper()
    print('-'*20)
    problem.Step(
        maxiter=200, 
        popsize=100, 
        elitek=0.2,
        tournk=0.8,
        tournsize=2,
        mutation=0.2, 
        sigma=0.25,
        crossover=0.4, 
        slack=1.12,
        # new_niche=1,
        disp_rate=20,
        )
    noptima, nfitness, nobjective, nfeasibility = problem.Terminate()
    for n in range(problem.nniche): 
        print(noptima[n], nfitness[n], nobjective[n], nfeasibility[n])
    PrintTimekeeper(on_switch=on_switch)
