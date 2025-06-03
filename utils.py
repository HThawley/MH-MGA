# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:24:29 2025

@author: hmtha
"""

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

