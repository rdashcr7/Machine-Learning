# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:38:59 2025

@author: RDASH
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#%% Preparing input and output files
in_dat = pd.read_excel(r"Modified KP sequences with z-scale.xlsx") #% Reading Input peptide sequences as z-scale numbers
in_dat.to_csv() #% converting to CSV

out_dat = pd.read_excel(r"700 modified KP sequences 13 Oct.xlsx") #% Reading output values from DPD simulation
out_dat.to_csv() #% converting to CSV
target_column = ['Mean Rg'] #% Defining output columns

#% Data splitting

# df[predictors] = df[predictors]/df[predictors].max()
X = in_dat.values #% input
y = out_dat[target_column].values #% Output

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

# Set up the Random Forest model
rf = RandomForestRegressor()

# Define the search space for the hyperparameters
search_space = {'max_depth': (1,50), 'n_estimators':(1,1000),'min_samples_split':(2,50),'min_samples_leaf':(2,50), 'max_features':[None, 'sqrt','log2'], 'bootstrap':[True, False]}


# Perform Bayesian optimization using BayesSearchCV
RF_model = BayesSearchCV(rf, search_space, n_iter=1000, cv=5, n_jobs=-1, random_state=6)

# Fit the model (this will start the optimization process)
RF_model.fit(X_train, y_train)
RF_pred_train = RF_model.predict(X_train)
RF_pred_test = RF_model.predict(X_test)

RF = r2_score(y_test,RF_pred_test)

with open("RF_bayesian_regression_R2_scores.txt", "w") as f:
    # write R2 scores to text file
    f.write("R2 score of training dataset is " + str(r2_score(y_train, RF_pred_train)))
    f.write("\n")
    f.write("MSE of training dataset is " + str(np.sqrt(mean_squared_error(y_train,RF_pred_train))))
    f.write("\n")
    f.write("R2 score of testing dataset is " + str(r2_score(y_test, RF_pred_test)))
    f.write("\n")
    f.write("MSE of testing dataset is " + str(np.sqrt(mean_squared_error(y_test,RF_pred_test))))
    f.write("\n")
    f.write("\n")

m_RF_train, b_RF_train = np.polyfit(y_train[:,0], RF_pred_train,1)
m_RF_test, b_RF_test = np.polyfit(y_test[:,0], RF_pred_test,1)
x_slope = np.linspace(6,13)

## Plot comparing predicted training dataset output with actual training dataset output
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(y_train[:,0], RF_pred_train, s=15, alpha=0.6, label = 'Prediction')
plt.plot(x_slope, x_slope, 'k-', lw=3, label = 'y=x')
plt.plot(x_slope, m_RF_train*x_slope+b_RF_train, 'r--', lw=3, label = 'Best linear fit')
plt.ylabel('Predicted <Rg>')
plt.xlabel('Observed <Rg>')
plt.title('Training Data')
plt.legend(loc=2, prop={'size':9})
plt.grid(True)
plt.tight_layout()

## Plot comparing predicted testing dataset output with actual testing dataset output
plt.subplot(122)
plt.scatter(y_test[:,0], RF_pred_test, s=15, alpha=0.6, label = 'Prediction')
plt.plot(x_slope, x_slope, 'k-', lw=3, label = 'y=x')
plt.plot(x_slope, m_RF_test*x_slope+b_RF_test, 'r--', lw=3, label = 'Best linear fit')
plt.ylabel('Predicted <Rg>')
plt.xlabel('Observed <Rg>')
plt.title('Testing Data')
plt.grid(True)
plt.tight_layout()
plt.grid(True)
plt.tight_layout()
plt.savefig('RF 700 modified KP Mean Rg Bayesian Regression.png', dpi = 1000)
