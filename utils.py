# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:24:29 2025

@author: hmtha
"""

import numpy as np 
from numba import njit

def islistlike(obj):
    if not hasattr(obj, "__iter__"):
        return False
    if not hasattr(obj, "__len__"):
        return False
    return True

def isfunclike(obj):
    if not hasattr(obj, "__call__"):
        return False
    return True

@njit 
def njit_deepcopy(new, old):
    flat_new = new.ravel()
    flat_old = old.ravel()
    
    for i in range(flat_old.shape[0]):
        flat_new[i] = flat_old[i]
        
