# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:10:16 2025

@author: hmtha
"""

# =============================================================================
# Evaluate termination criteria
# =============================================================================

class maxiter:
    """
    Maximum iterations termination criterion 
    """
    def __init__(self, maxiter):
        self.maxiter = maxiter
        self.iter = 0
        
    def __call__(self):
        self.iter += 1 
        if self.iter >= self.maxiter:
            return True
        return False
    
class convergence:
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        return False
    


