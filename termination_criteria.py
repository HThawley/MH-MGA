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

from datetime import datetime as dt
from datetime import timedelta as td
from time import perf_counter

class Convergence:
    """ 
    Base class for convergence criteria
    """
    def __init__(
            self, 
            *args, 
            **kwargs
            ):
        pass
    
class MultiConvergence(Convergence):
    """
    Template object for handling multiple convergence criteria
    
    Can be stacked to create complex convergence conditions
    """
    def __init__(
            self, 
            criteria: list[Convergence], 
            how: str = "or"
            ):
        assert how in ("and", "or")
        self.how = how
        self.ncrit = len(criteria)
        for i, crit in enumerate(criteria):
            assert isinstance(crit, Convergence), f"{i}: {crit}"
        self.criteria = criteria

            
    def __call__(
            self, 
            intermediate_result,
            ):
        nTrue = sum([crit(intermediate_result) for crit in self.criteria])
        
        if self.how == "and":
            convergence = nTrue == self.ncrit
        elif self.how == "or":
            convergence = nTrue >= 1 
        
        return convergence 
    
    def __repr__(self):
        return f"""Multi Criteria Convergence Object: 
    {self.ncrit} criteria with "{self.how}" logic. 
    Criteria: {[repr(crit) for crit in self.criteria]}"""

class Timeout(Convergence):
    """
    Terminates on reaching a pre-set value
    """
    def __init__(
            self, 
            timeout,
            start_attribute: str = None,
            time_attribute: str = None,
            timedelta_attribute: str = None,
            how = "dt"
            ):
        self.timeout = timeout

        assert how in ("dt", "perf"), f"`how` should be 'dt' (datetime.datetime.now) or 'perf' (time.perf_counter). Supplied {how}"
        self.now = dt.now if how == "dt" else perf_counter

        self.start = None
        self.start_attribute = start_attribute
        self.time_attribute = time_attribute
        self.timedelta_attribute = timedelta_attribute
        self.started = False
        
    def __call__(
            self, 
            intermediate_result = None,
            ):
        if self.timeout is None:
            return False
        
        if not self.started: 
            # timer should start at first call, not at init
            if self.start_attribute is not None: 
                self.start = getattr(intermediate_result, self.start_attribute)
            else: 
                self.start = self.now()
            self.started=True
            
        if self.timedelta_attribute is not None:
            time = getattr(intermediate_result, self.timedelta_attribute)
        elif self.time_attribute is not None:
            time = getattr(intermediate_result, self.time_attribute) - self.start
        else: 
            time = self.now() - self.start
                
        if time > self.timeout: 
            return True
        return False
    
    def __repr__(self):
        return f"""Timeout Convergence Criterion: {self.timeout}"""

class FixedValue(Convergence):
    """
    Terminates on reaching a pre-set value
    """
    def __init__(
            self, 
            target_value: float,
            maximize: bool,
            attribute: str = None, # named attribute of intermediate_result object
            ):
        self.target_value = target_value
        self.sense = 1.0 if maximize else -1.0
        self.attribute = attribute
        
    def __call__(
            self, 
            intermediate_result,
            ):
        value = getattr(intermediate_result, self.attribute) if self.attribute is not None else intermediate_result

        delta = self.sense * (value - self.target_value)
        if delta > 0: 
            return True
        return False
    
    def __repr__(self):
        if self.attribute is None: 
            return f"""Fixed Value Convergence Criterion: {self.target_value}"""
        else: 
            return f"""Fixed Value of '{self.attribute}' Convergence Criterion: {self.target_value}"""

class Maxiter(Convergence):
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
            intermediate_result = None,
            ):
        self.iter += 1 
        if self.iter >= self.maxiter:
            return True
        return False
    
    def __repr__(self):
        return f"""Maxiter Convergence Criterion: {self.iter} / {self.maxiter}"""

class ArithmeticStagnation(Convergence):
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
            maximize: bool = False,
            attribute: str = None, # named attribute of intermediate_result object
            ):
        self.i = 0
        self.niter = niter
        self.improvement = improvement
        self.sense = -1 if maximize else 1
        self.prev = self.sense * float('inf') 
        self.attribute = attribute
        
    def __call__(
            self, 
            intermediate_result
            ):
        value = getattr(intermediate_result, self.attribute) if self.attribute is not None else intermediate_result

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
        if self.attribute is None:
            return f"""
Arithemetic Stagnation Convergence Criterion: {self.niter} iterations 
with worse than {self.improvement} absolute improvement ({"maximizing" if self.sense == -1 else "minimizing"})"""
        else: 
            return f"""
Arithemetic Stagnation of '{self.attribute}' Convergence Criterion: {self.niter} iterations 
with worse than {self.improvement} absolute improvement ({"maximizing" if self.sense == -1 else "minimizing"})"""

class GradientStagnation(Convergence):
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
            maximize: bool = False,
            attribute: str = None, # named attribute of intermediate_result object
            ):
        assert window >= 2
        self.window = window
        self.improvement = improvement
        self.sense = -1 if maximize else 1
        self.prev = [self.sense * float('inf')]
        self.attribute = None
        
    def __call__(
            self, 
            intermediate_result
            ):
        value = getattr(intermediate_result, self.attribute) if self.attribute is not None else intermediate_result
        self.prev.append(value)
        if len(self.prev) <= self.window:
            return False
        
        delta = (self.prev.pop(0) - value) / self.window * self.sense
        
        if delta < self.improvement: 
            return True
        return False 
    
    def __repr__(self):
        if self.attribute is None:
            return f"""
Gradient Stagnation Convergence Criterion: {self.window} iterations 
with worse than {self.improvement} average improvement ({"maximizing" if self.sense == -1 else "minimizing"})"""
        else: 
            return f"""
Gradient Stagnation of '{self.attribute}' Convergence Criterion: {self.window} iterations 
with worse than {self.improvement} average improvement ({"maximizing" if self.sense == -1 else "minimizing"})"""
        
        
