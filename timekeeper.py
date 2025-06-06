
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:02:17 2025

@author: u6942852
"""

from datetime import datetime as dt
from datetime import timedelta as td
from numba import njit, objmode, float64, int64, boolean
from numba.experimental import jitclass
from numba.core.registry import CPUDispatcher
import numpy as np
import pandas as pd


spec = [
        ('on_switch', boolean),
        ('functions', int64[:]), 
        ('times', float64[:,:]),
        ('compile', float64[:,:]),
        ('calls', int64[:]),
        ('jitted', boolean[:]),
        ('_empty', boolean),
        ]

@jitclass(spec)
class Timekeeper:
    def __init__(self, on_switch=True):
        self.on_switch = on_switch
        if self.on_switch is False:
            return
        self.functions = np.zeros(1, dtype=np.int64)
        self.times = np.zeros((1, 3), dtype=np.float64)
        self.calls = np.zeros(1, dtype=np.int64)
        self.compile = np.zeros((1, 3), dtype=np.float64)
        self._empty=True
        with objmode():
            globals()['timekeeper_names'] = {}
    def Add_func(self, jit):
        if self._empty is True:
            self._empty = False
            self.jitted = np.array([jit], dtype=np.bool_)
            return 0
        self.functions = np.arange(len(self.functions)+1)
        self.times = np.vstack((self.times, np.zeros((1,3), np.float64)))
        self.compile = np.vstack((self.compile, np.zeros((1,3), np.float64)))
        self.calls = np.concatenate((self.calls, np.zeros(1, np.int64)))
        self.jitted = np.concatenate((self.jitted, np.array([jit], np.bool_)))
        return len(self.functions) - 1 
    def Update(self, index, time):
        if self.calls[index] == 0:
            self.compile[index] += time
        else: 
            self.times[index] += time
        self.calls[index] += 1
    
def PrintTimekeeper(path=None, console=True, combine=False):
    tk = globals()['timekeeper']
    if tk.on_switch is False:
        return
    names = globals()['timekeeper_names']
    if console is True:
        print("Timekeeper","="*50, sep='\n')
        lens = [[len(names[i]) for i in tk.functions], 
                [len(str(int(tk.calls[i]))) for i in tk.functions],
                [len(str(td(*tk.times[i]))) if tk.jitted[i]
                 else len(str(td(*tk.times[i]) + td(*tk.compile[i])))
                 for i in tk.functions]]
        maxlens = [max(_lens) for _lens in lens]
                   
        for i in tk.functions:
            if tk.jitted[i]: 
                print(
                    f'Function: {names[i]}{" "*(1+(maxlens[0]-lens[0][i]))}\t|'+
                    f'Calls: {int(tk.calls[i])}{" "*(1+(maxlens[1]-lens[1][i]))} \t|'+
                    f'Time: {td(*tk.times[i])}{" "*(1+(maxlens[2]-lens[2][i]))} \t|'+
                    f'Compile: {td(*tk.compile[i])}'
                      )
            else: 
                print(
                    f'Function: {names[i]}{" "*(1+(maxlens[0]-lens[0][i]))}\t|'+
                    f'Calls: {int(tk.calls[i])}{" "*(1+(maxlens[1]-lens[1][i]))}\t|'+
                    f'Time: {td(*tk.times[i])+td(*tk.compile[i])}{" "*(1+(maxlens[2]-lens[2][i]))} \t|'+
                     'Compile: -'
                      )
        print("="*50)
    if path is not None:
        indices = [names[index] for index in tk.functions]
            
        result = pd.DataFrame([], index=indices, 
                              columns=['calls', 'time'])
        for i, index in enumerate(indices):
            result.loc[index,:] = tk.calls[i], td(*tk.times[i])
        result.to_csv(path)

@njit 
def dt_now():
    now = np.empty(7, np.int64)
    with objmode():
        n = dt.now()
        now[:] = np.array([n.year, n.month, n.day, n.hour, n.minute, n.second, n.microsecond], np.int64)
    return now

@njit
def time_delta(start, end):
    delta = np.empty(3, np.float64)
    with objmode():
        d = dt(*end) - dt(*start)
        delta[:] = np.array([d.days, d.seconds, d.microseconds], np.float64)
    return delta

def keeptime(name=None):
    def decorator(func):
        jit = isinstance(func, CPUDispatcher)
        try: 
            tk = globals()['timekeeper']
        except KeyError:
            raise KeyError("Please declare an instance of timekeeper `timekeeper=Timekeeper()`")
        
        if tk.on_switch is False:
            return func
                    
        index = tk.Add_func(jit)
        
        if name is not None:
            try:
                globals()['timekeeper_names'][index]=name
            except KeyError:
                global timekeeper_names
                timekeeper_names = {}
                globals()['timekeeper_names'][index]=name
        
        if jit:
            @njit
            def wrapper(*args):
                start=dt_now()
                ret = func(*args)
                with objmode():
                    globals()['timekeeper'].Update(index, time_delta(start, dt_now()))
                return ret
        else:
            def wrapper(*args, **kwargs):
                start=dt_now()
                ret=func(*args, **kwargs)
                globals()['timekeeper'].Update(index, time_delta(start, dt_now()))
                return ret
        return wrapper
    return decorator

timekeeper=Timekeeper(on_switch=True)
# timekeeper_names={}

if __name__=='__main__':
    from time import sleep
    
    @keeptime('func1')
    @njit
    def func1(n=1_000_000):
        x = list(range(n))
        y = [y for y in x] 
    
    @keeptime('func2')
    @njit
    def func2(n=1_000_000):
        x = list(range(n))
        y = [y for y in x] 
    
    @keeptime('func3')
    @njit
    def func3(n=3):
        with objmode():
            sleep(0.3)
 
    @keeptime('func_rec')
    @njit
    def func_rec():
        func1()
        func3()
         
 
    @keeptime('func4')
    def func4():
        sleep(0.2)
        return
    
    func1()
    func1()
    func3()
    func2()
    func4()
    func_rec()
    PrintTimekeeper()
        
        
