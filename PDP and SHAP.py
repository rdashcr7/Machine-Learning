# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:37:09 2024

@author: rdash
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
target_column = ['Mean EtE'] #% Defining output columns

#% Data splitting

# df[predictors] = df[predictors]/df[predictors].max()
X = in_dat.values #% input
y = out_dat[target_column].values #% Output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6) #% splitting sample into training and testing samples

param_1 = {'max_depth': [None,10,20,30], 'n_estimators':[100,200,300,400],'min_samples_split':[10,20,25,40],'min_samples_leaf':[4,8,16,32], 'max_features':[None, 'sqrt','log2'], 'bootstrap':[True, False]}

RF_model = RandomForestRegressor(max_depth = 20, bootstrap = True, max_features = 'sqrt', min_samples_leaf = 4, min_samples_split = 10, n_estimators = 100)
# clf = GridSearchCV(estimator = RF_model, param_grid = param_1, cv =5, n_jobs = -1)

RF_model.fit(X_train,y_train.ravel())

#%% Feature Information
import_features = [8, 13, 18, 19, 33, 38, 39, 42, 59]
import_feature_names = ['Residue number 3 Electronic Nature', 'Residue number 5 Size', 'Residue number 7 Hydrophilicty', 
                        'Residue number 7 Size', 'Residue number 12 Hydrophilicty', 'Residue number 13 Electronic Nature',
                        'Residue number 14 Hydrophilicity', 'Residue number 15 Hydrophilicity', 'Residue number 20 Electronic Nature']
import_feature_zscale_no = ['z3', 'z2', 'z1', 'z2', 'z1', 'z3', 'z1', 'z1', 'z3']

#%% 1D Partial Dependence Plots (Seaborn Visualization)

# Loop through all important features
for i, feature_idx in enumerate(import_features):
    feature_name = import_feature_names[i]
    z_scale_val = import_feature_zscale_no[i]

    # Compute partial dependence values
    pd_result = partial_dependence(RF_model, X_train, [feature_idx], grid_resolution=100)
    
    # Extract grid and partial dependence values
    pd_values = pd_result.average[0]  # Average partial dependence values
    pd_grid = pd_result.grid_values[0]  # Grid points for the feature

    # Plot using Seaborn
    plt.figure(figsize=(60,60))
    sns.lineplot(x=pd_grid, y=pd_values, color='blue', lw=20)
    plt.title(f'Partial Dependence: {feature_name}', fontsize=150, fontweight='bold', fontname = 'arial')
    plt.xlabel(z_scale_val, fontsize=150, fontname = 'arial', fontweight='bold')
    plt.ylabel('Partial Dependence Value', fontsize=150, fontname = 'arial', fontweight='bold')
    plt.xticks(fontsize = 150, fontname = 'Arial', fontweight='bold')
    plt.yticks(fontsize = 150, fontname = 'Arial', fontweight='bold')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'Partial_Dependency_Plot_of_{import_feature_names[i]}.png', dpi = 500)
    plt.show()
    
#%% SHAP 3

# Create SHAP explainer (for a tree-based model, like RandomForest)
explainer = shap.TreeExplainer(RF_model, X_train)

# Calculate SHAP values (for regression, it is a numpy array)
shap_values = explainer.shap_values(X_train)

# Function to calculate interaction SHAP values
def calculate_interaction_value(model, X, feature_idx1, feature_idx2):
    # Extract the prediction for the baseline (f(S)) and various subsets of features
    baseline_prediction = model.predict(X)  # f(S)

    # Subset of X with feature_idx1
    X_with_feature1 = X.copy()
    X_with_feature1[:, feature_idx2] = 0  # Set feature idx2 to zero (i.e., S ∪ {i})
    prediction_with_feature1 = model.predict(X_with_feature1)  # f(S ∪ {i})

    # Subset of X with feature_idx2
    X_with_feature2 = X.copy()
    X_with_feature2[:, feature_idx1] = 0  # Set feature idx1 to zero (i.e., S ∪ {j})
    prediction_with_feature2 = model.predict(X_with_feature2)  # f(S ∪ {j})

    # Subset of X with both feature_idx1 and feature_idx2
    X_with_both_features = X.copy()
    prediction_with_both_features = model.predict(X_with_both_features)  # f(S ∪ {i,j})

    # Calculate the interaction SHAP value using the formula
    interaction_values = prediction_with_both_features - prediction_with_feature1 - prediction_with_feature2 + baseline_prediction
    
    return interaction_values

# Loop through pairs of important features
for idx1 in range(9):
    for idx2 in range(9):
        # Calculate the interaction by measuring the interaction SHAP values
        interaction_values = calculate_interaction_value(RF_model, X_train, import_features[idx1], import_features[idx2])
        
        # Calculate the maximum interaction value (you could adjust this threshold based on your data)
        max_interaction_value = np.max(interaction_values)
        
        # Only plot if the maximum interaction value is greater than 0.5
        if max_interaction_value > 1.5:
            # Create a dataframe for seaborn plot
            data = {
                f'SHAP Values for {import_feature_names[idx1]}': shap_values[:, import_features[idx1]], 
                f'SHAP Values for {import_feature_names[idx2]}': shap_values[:, import_features[idx2]],
                'Interaction SHAP Values': interaction_values  # Interaction SHAP values for coloring
            }

            df = pd.DataFrame(data)
            
            # Use Seaborn to create the scatter plot
            plt.figure(figsize=(80,60))
            scatter = sns.scatterplot(
                x=f'SHAP Values for {import_feature_names[idx1]}',
                y=f'SHAP Values for {import_feature_names[idx2]}',
                hue='Interaction SHAP Values',  # Color the points based on SHAP interaction values
                palette='viridis',  # Use the 'viridis' colormap
                data=df,
                s=5000,  # Marker size
                edgecolor='k',  # Black edge color for points
                legend=None  # Remove legend
            )
            
            # Add colorbar with Arial font
            norm = plt.Normalize(df['Interaction SHAP Values'].min(), df['Interaction SHAP Values'].max())  # Normalize the color scale
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])  # Empty array for the colorbar
            cbar = plt.colorbar(sm, ax=plt.gca())  # Add colorbar
            cbar.ax.tick_params(labelsize=150)
            for label in cbar.ax.get_yticklabels():  # Modify tick labels directly
                label.set_fontname('Arial')  # Set font type
                label.set_fontweight('bold')  # Set font size

            # Set labels and title with Arial font
            plt.xlabel(f'Shapley Values for {import_feature_names[idx1]}', fontsize=150, fontname='Arial', fontweight='bold')
            plt.ylabel(f'Shapley Values for {import_feature_names[idx2]}', fontsize=150, fontname='Arial', fontweight='bold')
            plt.xticks(fontsize = 150, fontname = 'Arial', fontweight='bold')
            plt.yticks(fontsize = 150, fontname = 'Arial', fontweight='bold')
            plt.title(f'SHAP Interaction',
                      fontsize=200, fontweight='bold', fontname='Arial')

            plt.grid(True)  # Enable grid
            plt.tight_layout()  # Improve spacing

            # Save the plot with high DPI
            plt.savefig(f'shap_interaction_{import_feature_names[idx1]}_{import_feature_names[idx2]}.png', dpi=500)
            plt.show()