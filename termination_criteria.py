# -*- coding: utf-8 -*-
"""
General purpose optimisation / MGA termination criteria. 

In general, these are classes which need to be instantiated. 
They should be called once during an iteration and return True if the convergence
    criteria is met. Otherwise return False
"""



class Maxiter:
    """
    Maximum iterations termination criterion 
    """
    def __init__(self, maxiter):
        self.maxiter = maxiter
        self.iter = -1
        
    def __call__(self):
        self.iter += 1 
        if self.iter >= self.maxiter:
            return True
        return False
    
    

class ArithmeticStagnation:
    """
    Convergence after a number of sequential iterations with below benchmark 
        improvement in optimum
    niter       - number of sequential iterations
    improvement - minimum improvement in optimum
    maximize    - whether improvement is positive/negative
    """
    def __init__(self, niter, improvement, maximize = False):
        self.i = 0
        self.niter = niter
        self.improvement = improvement
        self.sense = -1 if maximize else 1
        self.prev = self.sense * float('inf') 
        
    def __call__(self, value):
        delta = (self.prev - value) * self.sense
        if delta < self.improvement:
            self.i += 1 
        else: 
            self.i = 0 
        if self.i >= self.niter: 
            return True
        return False 
    
class GradientStagnation:
    """
    Convergence after a number of sequential iterations with average improvment 
        below a benchmark
    window      - number of sequential iterations
    improvement - minimum average improvement in optimum
    maximize    - whether improvement is positive/negative
    """

    def __init__(self, window, improvement, maximize = False):
        assert window >= 2
        self.window = window
        self.improvement = improvement
        self.sense = -1 if maximize else 1
        self.prev = [self.sense * float('inf')]
        
    def __call__(self, value):
        self.prev.append(value)
        if len(self.prev) <= self.window:
            return False
        
        delta = (self.prev.pop(0) - value) / self.window * self.sense
        
        if delta < self.improvement: 
            return True
        return False 
        
        
