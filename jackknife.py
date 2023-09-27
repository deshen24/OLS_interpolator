"""
This code replicates Tables 2-3 of the paper, Algebraic and Statistical Properties of the Ordinary Least Squares Interpolator.

Contributors: 
- Dennis Shen (dennis.shen@marshall.usc.edu)
- Dogyoon Song (dogyoons@umich.edu)
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# check user input
if len(sys.argv)!=2:
    print('Usage: python {} <alpha>'.format(sys.argv[0]))
    sys.exit()
alpha = float(sys.argv[1])

print("Alpha: {:.2f}".format(alpha))

#-----------------
# interval length
#-----------------
def interval_len(ub, lb):
    return ub-lb

#-----------
# Jackknife 
#-----------
def jackknife(y_pred, errs, alpha=0.1): 
    
    # take absolute value of errors
    errs_abs = np.abs(errs)
    
    # compute (1-alpha) quantile
    r_alpha = np.quantile(a=errs_abs, q=1-alpha)
    
    # compute lower and upper bounds
    lb = y_pred - r_alpha
    ub = y_pred + r_alpha
    return (lb, ub)

#------------
# Jackknife+ 
#------------
def jackknifePlus(pred_loo, errs, alpha=0.1): 
    
    # take absolute value of errors
    errs_abs = np.abs(errs)
    
    # compute index
    idx = int(np.ceil((1-alpha) * (n+1))) 
    
    # compute upper quantile
    ub = np.sort(pred_loo + errs_abs)[idx-1]
    
    # compute lower quantile
    lb = -np.sort(-pred_loo + errs_abs)[idx-1]
    return (lb, ub)

#------------------
# LOO computations
#------------------
# residuals
def compute_res_loo(X, y): 
    Hx = np.linalg.inv(X@X.T)
    Dx = np.diag(np.diag(Hx))
    return np.linalg.inv(Dx) @ Hx @ y 

# predictions
def compute_pred_loo(X_pinv, y_pred, res_loo, x_test):
    a = y_pred * np.ones(res_loo.shape[0])
    b = np.diag(res_loo) @ X_pinv.T @ x_test
    return a - b 

#------------
# Simulation 
#------------
# parameters
p = 200
sample_sizes = np.linspace(25, p, 8)
sample_sizes = [int(n) for n in sample_sizes]
n_iters = 10000
signal_std = 1 
noise_std = 1

# initialize 
algs = ['jackknife', 'jackknife+']
coverage_dict = {alg: {n: 0 for n in sample_sizes} for alg in algs}
interval_len_dict = {alg: {n: np.zeros(n_iters) for n in sample_sizes} for alg in algs}

# construct underlying regression model  
beta = np.ones(p) / np.sqrt(p)

# iterate through different sample sizes
for n in sample_sizes:

    print("========================")
    print("n = {}".format(n))

    # set random seed 
    np.random.seed(0)
    
    # repeat simulations
    for i in range(n_iters): 
        
        """ 
        Generate data 
        """
        # generate covariates
        X = np.random.normal(size=(n, p), scale=signal_std)
        X_pinv = np.linalg.pinv(X)
        x_test = np.random.normal(size=p, scale=signal_std)
        
        # generate responses
        y_train = (X @ beta) + np.random.normal(size=n, scale=noise_std)
        y_test = np.dot(x_test, beta) + np.random.normal(scale=noise_std)
        
        """
        Estimation
        """
        # compute regession model
        beta_hat = X_pinv @ y_train 
        
        # compute prediction value
        y_pred = np.dot(x_test, beta_hat)
        
        # compute LOO residuals
        res_loo = compute_res_loo(X, y_train)
        
        # compute LOO predictions 
        pred_loo = compute_pred_loo(X_pinv, y_pred, res_loo, x_test)

        """
        Inference
        """
        for alg in algs: 
            if alg=='jackknife': 
                (lb, ub) = jackknife(y_pred, res_loo, alpha=alpha)
            elif alg=='jackknife+':
                (lb, ub) = jackknifePlus(pred_loo, res_loo, alpha=alpha)
            if (y_test >= lb) and (y_test <= ub): 
                coverage_dict[alg][n] += 1
            interval_len_dict[alg][n][i] = interval_len(ub, lb)
            
    # report inference results 
    for alg in algs: 
        coverage_dict[alg][n] /= n_iters 
        print("------------------------")
        print("*** {} ***".format(alg))
        print("coverage = {:.3f}".format(coverage_dict[alg][n]))
        print("average interval len = {:.3f}".format(interval_len_dict[alg][n].mean()))
        print() 

