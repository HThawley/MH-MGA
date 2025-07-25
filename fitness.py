# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 19:41:57 2025

@author: u6942852
"""

from numba import njit
import numpy as np 


@njit
def euclideanDistance(p1, p2):
    """Euclidean distance"""
    return sum((p1-p2)**2)**0.5
    
@njit
def pointwiseDistance(p1, p2):
    return min(np.abs(p1-p2))

