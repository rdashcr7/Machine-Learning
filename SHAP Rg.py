# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:06:22 2024

@author: RDASH
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import seaborn as sns
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from itertools import combinations
import shap
from itertools import combinations

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6) #% splitting sample into training and testing samples


RF_model = RandomForestRegressor(max_depth = 30, bootstrap = True, max_features = 0.5, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 100)
RF_model.fit(X_train,y_train.ravel())


#%% Feature Information
import_features = [8, 13, 18, 19, 33, 38, 39, 42, 59]
import_feature_names = ['Residue number 3 Electronic Nature', 'Residue number 5 Size', 'Residue number 7 Hydrophilicity', 
                        'Residue number 7 Size', 'Residue number 12 Hydrophilicity', 'Residue number 13 Electronic Nature',
                        'Residue number 14 Hydrophilicity', 'Residue number 15 Hydrophilicity', 'Residue number 20 Electronic Nature']
import_feature_zscale_no = ['z3', 'z2', 'z1', 'z2', 'z1', 'z3', 'z1', 'z1', 'z3']

#%% SHAP
 
explainer = shap.TreeExplainer(RF_model, X_train)
 
shap_values = explainer.shap_values(X_train)
 
def calculate_interaction_value(model, X, feature_idx1, feature_idx2):

    baseline_prediction = model.predict(X) 

    X_with_feature1 = X.copy()
    X_with_feature1[:, feature_idx2] = 0  # Set feature idx2 to zero (i.e., S ∪ {i})
    prediction_with_feature1 = model.predict(X_with_feature1)  

    X_with_feature2 = X.copy()
    X_with_feature2[:, feature_idx1] = 0  # Set feature idx1 to zero
    prediction_with_feature2 = model.predict(X_with_feature2) 


    X_with_both_features = X.copy()
    prediction_with_both_features = model.predict(X_with_both_features)  

    interaction_values = prediction_with_both_features - prediction_with_feature1 - prediction_with_feature2 + baseline_prediction
    
    return interaction_values
 
for idx1 in range(9):
    for idx2 in range(9):

        interaction_values = calculate_interaction_value(RF_model, X_train, import_features[idx1], import_features[idx2])
        
        max_interaction_value = np.max(interaction_values)
        

        if max_interaction_value > 1:

            data = {
                f'SHAP Values for {import_feature_names[idx1]}': shap_values[:, import_features[idx1]], 
                f'SHAP Values for {import_feature_names[idx2]}': shap_values[:, import_features[idx2]],
                'Interaction SHAP Values': interaction_values  
            }

            df = pd.DataFrame(data)
            

            plt.figure(figsize=(80,60))
            scatter = sns.scatterplot(
                x=f'SHAP Values for {import_feature_names[idx1]}',
                y=f'SHAP Values for {import_feature_names[idx2]}',
                hue='Interaction SHAP Values', 
                palette='viridis', 
                data=df,
                s=5000, 
                edgecolor='k', 
                legend=None  
            )
            

            norm = plt.Normalize(df['Interaction SHAP Values'].min(), df['Interaction SHAP Values'].max())
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])  
            cbar = plt.colorbar(sm, ax=plt.gca()) 
            cbar.ax.tick_params(labelsize=150)
            for label in cbar.ax.get_yticklabels():
                label.set_fontname('Arial')  
                label.set_fontweight('bold')  


            plt.xlabel(f'Shapley Values for {import_feature_names[idx1]}', fontsize=150, fontname='Arial', fontweight='bold')
            plt.ylabel(f'Shapley Values for {import_feature_names[idx2]}', fontsize=150, fontname='Arial', fontweight='bold')
            plt.xticks(fontsize = 150, fontname = 'Arial', fontweight='bold')
            plt.yticks(fontsize = 150, fontname = 'Arial', fontweight='bold')
            plt.title(f'SHAP Interaction',
                      fontsize=200, fontweight='bold', fontname='Arial')

            plt.grid(True)  
            plt.tight_layout()


            plt.savefig(f'shap_interaction_{import_feature_names[idx1]}_{import_feature_names[idx2]}.png', dpi=500)
            plt.show()

