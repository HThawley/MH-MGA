# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:22:27 2025

@author: u6942852
"""

import numpy as np
import pandas as pd
import chaospy as cp
from datetime import datetime as dt
from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from numba import njit, prange
import json


import os
import pickle
from time import perf_counter
#%%

np.set_printoptions(suppress=True)

class PCEmodel:
    def __init__(
            self, 
            metadata_path: str=None,
            ):
        self._is_trained = False
        
        if metadata_path is not None:
            self.load_model(metadata_path)
            
        else: 
            self.coefficients = None
            self.polynomial_order = None
            self.method = None
            self.scaler_mean = None
            self.scaler_scale = None

            self.num_inputs = 0
            
    def _data_assertions(self, input, output):
        assert isinstance(input, np.ndarray), "input should be 2d numpy array"
        assert input.ndim == 2, "input should be 2d numpy array"
        assert isinstance(output, np.ndarray), "output should be 1d numpy array"
        assert output.ndim == 1, "output should be 1d numpy array"
        assert input.shape[1] == self.num_inputs
        assert input.shape[0] == output.shape[0], "input (N, M) and output (N,) shapes should match"
        assert ((input - self.ub) < 0.001).all(), "input does not obey supplied bounds"
        assert ((self.lb - input) < 0.001).all(), "input does not obey supplied bounds"
    
    def _create_scaler(
            self, 
            input,
            ):
        scaler = StandardScaler()
        if self.scaler_mean is not None and self.scaler_scale is not None:
            scaler.mean_ = self.scaler_mean
            scaler.scale_ = self.scaler_scale
            scaler.n_features_in_ = len(self.scaler_mean)
        else: 
            scaler.fit(input)
            self.scaler_mean = scaler.mean_ 
            self.scaler_scale = scaler.scale_ 
        return scaler
    
    def preprocess(
            self, 
            input, 
            ):
        # input = normalize(input, self.lb, self.ub)
        scaler = self._create_scaler(input)
        # input = scaler.transform(input)
        return input
        
    def train(
            self, 
            input, 
            output, 
            bounds, 
            polynomial_order=2, 
            verbose=True, 
            method="lars",
            ):
        self.lb, self.ub = bounds
        assert len(self.lb) == len(self.ub)
        self.num_inputs = len(self.lb)
        self._data_assertions(input, output)
        
        assert isinstance(polynomial_order, int)
        assert polynomial_order > 1
        
        self.method = method
        self.polynomial_order = polynomial_order
        
        if self.method == "lars":
            method = lm.Lars(fit_intercept=False)
        
        start = dt.now()
        if verbose:
            print("Starting training:", start)
            print("Preprocessing training data... | Time:", dt.now())
        input = self.preprocess(input)
        joint_distribution = cp.J(*[cp.Uniform(0,1) for _ in range(self.num_inputs)])
        if verbose: 
            print("Creating polynomial basis...   | Time:", dt.now())
        polynomial_basis = cp.expansion.stieltjes(self.polynomial_order, joint_distribution)
        if verbose: 
            print("Fitting Model...               | Time:", dt.now())
        self.model = cp.fit_regression(
            polynomials=polynomial_basis,
            abscissas=input.T, 
            evals=output,
            model=lm.Lars(fit_intercept=False), 
            )
        self._is_trained = True
        if verbose: 
            print("Finished Succesfully.          | Time:", dt.now())
            print("Took:", dt.now() - start)

    def predict(
            self, 
            input,
            ):
        self.preprocess(input)
        return self.model(*input.T)
        
    def score(
            self, 
            true_output,
            predicted_output,
            metric="mean_poisson_deviance",
            ):
        assert metric in ("mean_poisson_deviance", "mean_squared_error")
        if metric == "mean_poisson_deviance":
            return mean_poisson_deviance(true_output, predicted_output)
        else: # metric == "mean_squared_error"
            return mean_squared_error(true_output, predicted_output)
        
    def save_model(
            self,
            filepath:str,
            overwrite = False,
            ):
        assert self._is_trained, "Cannot save an untrained model"
        metadata = {
            "polynomial_order" : self.polynomial_order,
            "method" : self.method,
            "num_inputs" : self.num_inputs,
            "scaler_mean" : self.scaler_mean.tolist(),
            "scaler_scale" : self.scaler_scale.tolist(),
            "lb" : self.lb.tolist(),
            "ub" : self.ub.tolist(),
            "exponents" : self.model.exponents.tolist(),
            "coefficients" : self.model.coefficients,
            "names" : self.model.names,
            }
        for k, v in metadata.items():
            assert v is not None, f"Cannot save an untrained model. ({k} is None)"
        if overwrite is False:
            if os.path.exists(filepath):
                os.mkdir("tmp_model_save")
                with open("tmp.json", "w") as f:
                    json.dump(metadata, f, indent=4)
                raise Exception(
"""Cannot overwrite existing saved model. Pass "`overwrite = True` or 
remove existing file. Current model saved as "tmp.json" """)
        with open(filepath+".json", "w") as f:
            json.dump(metadata, f, indent=4)

    
    def load_model(
            self, 
            filepath,
            verbose=True,
            ):
        start = dt.now()
        if verbose: 
            print("Reading save file... | Time:", dt.now())
        with open(filepath+".json", "r") as f:
            metadata = json.load(f)
    
        self.polynomial_order = metadata.get("polynomial_order")
        self.method = metadata.get("method")
        self.num_inputs = metadata.get("num_inputs")
        self.scaler_mean = np.array(metadata.get("scaler_mean"))
        self.scaler_scale = np.array(metadata.get("scaler_scale"))
        self.lb = np.array(metadata.get("lb"))
        self.ub = np.array(metadata.get("ub"))
        
        exponents = np.array(metadata.get("exponents"))
        coefficients = np.array(metadata.get("coefficients"))
        names = tuple(metadata.get("names"))
        
        if verbose: 
            print("Creating Model...     | Time:", dt.now())
            
        self.model = cp.polynomial_from_attributes(
            exponents = exponents,
            coefficients = coefficients,
            names = names,
            )
        self._is_trained=True
        if verbose: 
            print("Finished Succesfully. | Time:", dt.now())
            print("Took:", dt.now() - start)
    
@njit
def normalize(data, lb, ub):
    """ data.shape[0] == len(lb) == len(ub) """
    data = (data - lb) / (ub - lb)
    return data
    
@njit
def rmse(arr1, arr2):
    return np.mean((arr1-arr2)**2)**0.5

if __name__=="__main__":


    # @njit
    # def Objective(values_array): 
    #     """ Vectorized = True """
    #     z = 2 + np.zeros(values_array.shape[0], np.float64)
    #     for i in range(values_array.shape[0]):
    #         for j in range(values_array.shape[1]):
    #             z[i] += np.sin(19 * np.pi * values_array[i, j]) + values_array[i, j] / 1.7
    #     return z   
    
    @njit
    def Objective(values): 
        """ Vectorized = False """
        z = 2
        for i in range(values.shape[0]):
            z += np.sin(19 * np.pi * values[i]) + values[i] / 1.7
        return z   
    
    lb = np.zeros(2, float)
    ub = np.ones(2, float)
    
    # data = pd.read_csv(
    #     ..., 
    #     # skiprows = 4_000_000,
    #     # nrows=20_000, 
    #     header=None
    #     ).to_numpy()
    
    # rng = np.random.default_rng(seed=1)
    # rng.shuffle(data)
    # output_data = data[:, 0]
    # input_data = data[:, 1:]
    # del data
    
    @njit(parallel=True)
    def random_sample(func, lb, ub, rng, n=100_000):
        ndim = len(lb)
        ret_input = np.empty((n, ndim), np.float64)
        ret_output= np.empty(n, np.float64)
        # random seeding won't work in parallel
        for i in prange(n):
            for j in range(ndim):
                ret_input[i, j] = rng.uniform(lb[j], ub[j])
            ret_output[i] = func(ret_input[i])
        return ret_input, ret_output
    
    rng = np.random.default_rng()
    input_data, output_data = random_sample(Objective, lb, ub, rng, 1_000_000)
    
    cutoff = int(0.95*len(output_data))
    
    test_output = output_data[cutoff:]
    test_input = input_data[cutoff:, :]
    
    train_output = output_data[:cutoff]
    train_input = input_data[:cutoff, :]
    
    # if os.path.exists("pce.json"):
    if False:
        model = PCEmodel("pce")
    else:
        model = PCEmodel()
        model.train(
            train_input, 
            train_output, 
            (lb, ub), 
            polynomial_order = 6,
            )
        
        model.save_model("pce")

    pred_train_output = model.predict(train_input)
    try: 
        train_score = model.score(train_output, pred_train_output)
    except: 
        train_score = np.nan
    train_mse = model.score(train_output, pred_train_output, "mean_squared_error")
    train_rmse = rmse(train_output, pred_train_output)
    
    pred_test_output = model.predict(test_input)
    try: 
        test_score = model.score(test_output, pred_test_output)
    except: 
        test_score = np.nan
    test_mse = model.score(test_output, pred_test_output, "mean_squared_error")
    test_rmse = rmse(test_output, pred_test_output)
    
    print(f"""
    Poisson deviation:
        score on training dataset: {train_score} / 1.0
        score on testing dataset: {test_score} / 1.0
    mean squared error:
        score on training dataset: {train_mse} / 1.0
        score on testing dataset: {test_mse} / 1.0
    raw rmse: 
        score on training dataset: {train_rmse}
        score on testing dataset: {test_rmse}
        mean of (test + train) outputs: {np.mean(output_data)}
        """)
        
