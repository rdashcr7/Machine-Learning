# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:06:22 2024

@author: RDASH
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
import seaborn as sns
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from itertools import combinations

#%% Preparing input and output files
in_dat = pd.read_excel(r"Modified KP sequences with z-scale.xlsx") 
in_dat.to_csv() 

out_dat = pd.read_excel(r"700 modified KP sequences 13 Oct.xlsx") 
out_dat.to_csv()
target_column = ['Mean Rg']

#% Data splitting

# df[predictors] = df[predictors]/df[predictors].max()
X = in_dat.values #% input
y = out_dat[target_column].values #% Output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6) #% splitting sample into training and testing samples


RF_model = RandomForestRegressor(max_depth = 30, bootstrap = True, max_features = 0.5, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 100)
RF_model.fit(X_train,y_train.ravel())

#%% partial dependency plots

z1_range = np.arange(-4.92, 100, 3.64)
z2_range = np.arange(-5.36, 100, 3.65)
z3_range = np.arange(-3.44, 100, 4.13)

import_features = [8, 13, 18, 19, 33, 38, 39, 42, 59]
import_feature_names = ['Residue number 3 z3', 'Residue number 5 z2', 'Residue number 7 z1', 'Residue number 7 z2', 'Residue number 12 z1', 'Residue number 13 z3', 'Residue number 14 z1', 'Residue number 15 z1', 'Residue number 20 z3' ]


# Plot 1D Partial Dependence for each feature
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    RF_model, X_train, import_features, 
    grid_resolution=50, ax=ax
)
plt.suptitle('Partial Dependence Plots for Important Features', fontsize=16, fontweight='bold')
plt.show()

#%% partial dependency of two features

feature_pairs = list(combinations(import_features, 2)) 

for pair in feature_pairs:
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        RF_model, X_train, [pair],  # Pass each pair as a tuple
        grid_resolution=50, ax=ax
    )
    plt.suptitle(f'2D Partial Dependence Plot for Features {pair[0]} and {pair[1]}', 
                  fontsize=16, fontweight='bold')
    plt.show()

