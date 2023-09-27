"""
This code replicates Figures 1-3 of the paper, Algebraic and Statistical Properties of the Ordinary Least Squares Interpolator.

Contributors: 
- Dennis Shen (dennis.shen@marshall.usc.edu)
- Dogyoon Song (dogyoons@umich.edu)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# set random seed 
np.random.seed(0)

# compuate variance estimator based on LOO residuals
def compute_loo_var(X, y): 
    G = np.linalg.inv(X@X.T)
    D = np.diag(np.diag(G))
    Prod = np.linalg.inv(D) @ G
    res_loo = Prod @ y 
    tracex = np.trace(Prod @ Prod.T)
    return (np.linalg.norm(res_loo)**2) / tracex 

#------------------------------------
# Simulation I: Fixed p increasing n
#------------------------------------
print("====================")
print("*** Simulation I ***")
print("====================")

# parameters
p = 200
sample_sizes = np.linspace(int(0.125*p), int(0.875*p), 7)
sample_sizes = [int(n) for n in sample_sizes]
n_iters = 10000
signal_std = 1
noise_std = 1
noise_types = ['Gaussian', 'Laplacian']

print("p = {}".format(p))
print()

# initialize
noise_var = noise_std ** 2

# iterate through noise types 
for noise_type in noise_types:
 
    print("Noise type: {}".format(noise_type))

    # construct underlying regression model  
    beta = np.ones(p) / np.sqrt(p)

    # initialize variance estimates
    var_est_dict = {n: np.zeros(n_iters) for n in sample_sizes}

    # iterate through different sample sizes
    for n in sample_sizes: 
        
        print("n = {}...".format(n))
        
        # repeat simulations
        for i in range(n_iters): 

            # generate underlying data
            X = np.random.normal(size=(n, p), scale=signal_std)
            
            # generate responses
            if noise_type=='Gaussian':
                y = (X @ beta) + np.random.normal(size=n, scale=noise_std)
            elif noise_type=='Laplacian':
                y = (X @ beta) + np.random.laplace(size=n, scale=noise_std)
            
            # variance estimate
            var_est_dict[n][i] = compute_loo_var(X, y) 

    # store in dataframe 
    df_var = pd.DataFrame(columns=sample_sizes)
    for n in sample_sizes:
        df_var[n] = var_est_dict[n] - noise_var
        
    # Plot
    fname = os.path.join(output_dir, "bias_sim1_{}".format(noise_type))
    plt.figure(dpi=150, figsize=(9,6))
    plt.grid()
    sns.violinplot(data=df_var)
    plt.axhline(y=0, linestyle='--', color='black')
    plt.ylim(-2, 12)
    plt.title('{} noise: fixed $p$ with increasing $n$'.format(noise_type))
    plt.ylabel('Bias ($\widehat{\sigma}^2 - \sigma^2$)')
    plt.xlabel('Sample size ($n$)')
    plt.savefig(fname, dpi=150, figsize=(9,6), bbox_inches="tight")
    plt.close()
    print()

#--------------------------------------------------
# Simulation II: Fixed ratio increasing dimensions
#--------------------------------------------------
print("=====================")
print("*** Simulation II ***")
print("=====================")

# parameters
p_sizes = np.linspace(200, 2000, 10)
p_sizes = [int(p) for p in p_sizes]
n_iters = 10000
signal_std = 1
noise_std = 1
np_ratio = 0.5

# initialize 
var_est_dict = {p: np.zeros(n_iters) for p in p_sizes}
noise_var = noise_std ** 2

print("ratio = {:.2f}".format(n/p))

# iterate through diff num. covariates
for p in p_sizes: 
    
    print('p = {}...'.format(p))
    
    # determine sample size
    n = int(np_ratio*p)
    
    # construct underlying regression model  
    beta = np.ones(p) / np.sqrt(p)
    
    # repeat simulations
    for i in range(n_iters): 
    
        # generate underlying data
        X = np.random.normal(size=(n, p), scale=signal_std)
        
        # generate responses
        y = (X @ beta) + np.random.normal(size=n, scale=noise_std)
        
        # variance estimate
        var_est_dict[p][i] = compute_loo_var(X, y)

# store in dataframe 
df_var = pd.DataFrame(columns=p_sizes)
for p in p_sizes:
    df_var[p] = var_est_dict[p] - noise_var
    
# Plot
fname = os.path.join(output_dir, "bias_sim2")
plt.figure(dpi=150, figsize=(9,6))
plt.grid()
sns.violinplot(data=df_var)
plt.axhline(y=0, linestyle='--', color='black')
plt.ylim(-1, 4)
plt.title('Fixed $(n,p)$ with increasing dimensions')
plt.ylabel('Bias ($\widehat{\sigma}^2 - \sigma^2$)')
plt.xlabel('Covariate size ($p$)')
plt.savefig(fname, dpi=150, figsize=(9,6), bbox_inches="tight")
plt.close()
print()

#-------------------------------------------------------
# Simulation III: Fixed ratio increasing noise variance
#-------------------------------------------------------
print("======================")
print("*** Simulation III ***")
print("======================")

# parameters
p = 200
n = 100
n_iters = 10000
signal_std = 1
noise_stds = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# initialize variance estimates
var_est_dict = {noise_std: np.zeros(n_iters) for noise_std in noise_stds}

print("ratio = {:.2f}".format(n/p))
 
# construct underlying regression model  
beta = np.ones(p) / np.sqrt(p)

# iterate through diff SNRs
for noise_std in noise_stds: 

    print('Noise std = {}...'.format(noise_std))
    
    # repeat simulations
    for i in range(n_iters): 
    
        # generate underlying data
        X = np.random.normal(size=(n, p), scale=signal_std)
        
        # generate responses
        y = (X @ beta) + np.random.normal(size=n, scale=noise_std)
        
        # variance estimate
        var_est_dict[noise_std][i] = compute_loo_var(X, y)

# store in dataframe 
df_var = pd.DataFrame(columns=noise_stds)
for noise_std in noise_stds:
    noise_var = noise_std ** 2
    df_var[noise_std] = var_est_dict[noise_std] - noise_var
    
# Plot
fname = os.path.join(output_dir, "bias_sim3")
plt.figure(dpi=150, figsize=(9,6))
plt.grid()
sns.violinplot(data=df_var)
plt.axhline(y=0, linestyle='--', color='black')
plt.ylim(-2, 6)
plt.title('Fixed $(n,p)$ with increasing noise variance')
plt.ylabel('Bias ($\widehat{\sigma}^2 - \sigma^2$)')
plt.xlabel('Noise standard deviation ($\sigma$)')
plt.savefig(fname, dpi=150, figsize=(9,6), bbox_inches="tight")
plt.close()

fname = os.path.join(output_dir, "bias_sim3_zoom")
plt.figure(dpi=150, figsize=(9,6))
plt.grid()
sns.violinplot(data=df_var[noise_stds[:5]])
plt.axhline(y=0, linestyle='--', color='black')
plt.ylim(-2, 6)
plt.title('Fixed $(n,p)$ with increasing noise variance')
plt.ylabel('Bias ($\widehat{\sigma}^2 - \sigma^2$)')
plt.xlabel('Noise standard deviation ($\sigma$)')
plt.savefig(fname, dpi=150, figsize=(9,6), bbox_inches="tight")
plt.close()
