# -*- coding: utf-8 -*-
"""
General purpose optimisation / MGA termination criteria. 

These are classes which need to be instantiated. 
They should be called once during an iteration and return True if the convergence
    criteria is met. Otherwise return False
    
Possible future convergence criteria
    - standard deviation of objective function values (as in scipy's implementation of DE)
    - VESA inspired colume additions of new noptima
        - questions remain about whether this is applicable if we are not aiming to identify extrema
    - ... 
    
"""

class ConvergenceCriterion:
    """ 
    Base class for convergence criteria
    """
    def __init__(
            self, 
            *args, 
            **kwargs
            ):
        pass
    
class MultiCriteriaConvergence(ConvergenceCriterion):
    """
    Template object for handling multiple convergence criteria
    
    Can be stacked to create complex convergence conditions
    """
    def __init__(
            self, 
            criteria: list[ConvergenceCriterion], 
            how="or"
            ):
        assert how in ("and", "or")
        self.how = how
        self.ncrit = len(criteria)
        for i, crit in enumerate(criteria):
            assert isinstance(crit, ConvergenceCriterion), f"{i}: {crit}"
        self.criteria = criteria

            
    def __call__(
            self, 
            *args, 
            **kwargs
            ):
        nTrue = sum([crit(*args, **kwargs) for crit in self.criteria])
        
        if self.how == "and":
            convergence = nTrue == self.ncrit
        elif self.how == "or":
            convergence = nTrue >= 1 
        
        return convergence 
    
    def __repr__(self):
        return f"""Multi Criteria Convergence Object: 
    {self.ncrit} criteria with "{self.how}" logic. 
    Criteria: {[repr(crit) for crit in self.criteria]}"""
            
class Maxiter(ConvergenceCriterion):
    """
    Maximum iterations termination criterion 
    """
    def __init__(
            self, 
            maxiter: int,
            ):
        self.maxiter = maxiter
        self.iter = -1
        
    def __call__(
            self, 
            *args, 
            **kwargs
            ):
        self.iter += 1 
        if self.iter >= self.maxiter:
            return True
        return False
    
    def __repr__(self):
        return f"""Maxiter Convergence Criterion: {self.iter} / {self.maxiter}"""

class ArithmeticStagnation(ConvergenceCriterion):
    """
    Convergence after a number of sequential iterations with below benchmark 
        improvement in optimum
    niter       - number of sequential iterations
    improvement - minimum improvement in optimum
    maximize    - whether improvement is positive/negative
    """
    def __init__(
            self,
            niter: int, 
            improvement: int|float, 
            maximize: bool = False
            ):
        self.i = 0
        self.niter = niter
        self.improvement = improvement
        self.sense = -1 if maximize else 1
        self.prev = self.sense * float('inf') 
        
    def __call__(
            self, 
            value: int, 
            *args, 
            **kwargs
            ):
        delta = (self.prev - value) * self.sense
        self.prev = value
        if delta < self.improvement:
            self.i += 1 
        else: 
            self.i = 0 
        if self.i >= self.niter: 
            return True
        return False 
    
    def __repr__(self):
        return f"""Arithemetic Stagnation Convergence Criterion: {self.niter} iterations 
    with worse than {self.improvement} improvement ({"maximizing" if self.sense == -1 else "minimizing"})"""
    
class GradientStagnation(ConvergenceCriterion):
    """
    Convergence after a number of sequential iterations with average improvment 
        below a benchmark
    window      - number of sequential iterations
    improvement - minimum average improvement in optimum
    maximize    - whether improvement is positive/negative
    """

    def __init__(
            self, 
            window: int, 
            improvement: int|float, 
            maximize: bool = False
            ):
        assert window >= 2
        self.window = window
        self.improvement = improvement
        self.sense = -1 if maximize else 1
        self.prev = [self.sense * float('inf')]
        
    def __call__(
            self, 
            value: int, 
            *args, 
            **kwargs
            ):
        self.prev.append(value)
        if len(self.prev) <= self.window:
            return False
        
        delta = (self.prev.pop(0) - value) / self.window * self.sense
        
        if delta < self.improvement: 
            return True
        return False 
    
    def __repr__(self):
        return f"""Gradient Stagnation Convergence Criterion: {self.window} iterations 
    with worse than {self.improvement} average improvement ({"maximizing" if self.sense == -1 else "minimizing"})"""
        
        
