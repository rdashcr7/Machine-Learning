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

# #%% partial dependency plots

# z1_range = np.arange(-4.92, 100, 3.64)
# z2_range = np.arange(-5.36, 100, 3.65)
# z3_range = np.arange(-3.44, 100, 4.13)

# import_features = [8, 13, 18, 19, 33, 38, 39, 42, 59]
# import_feature_names = ['Residue number 3 z3', 'Residue number 5 z2', 'Residue number 7 z1', 'Residue number 7 z2', 'Residue number 12 z1', 'Residue number 13 z3', 'Residue number 14 z1', 'Residue number 15 z1', 'Residue number 20 z3' ]


# # Plot 1D Partial Dependence for each feature
# fig, ax = plt.subplots(figsize=(12, 8))
# PartialDependenceDisplay.from_estimator(
#     RF_model, X_train, import_features, 
#     grid_resolution=50, ax=ax
# )
# plt.suptitle('Partial Dependence Plots for Important Features', fontsize=16, fontweight='bold')
# plt.show()

# #%% partial dependency of two features

# feature_pairs = list(combinations(import_features, 2))  # Generate all pairs of features

# # Loop to generate 2D PDP for each pair of features
# for pair in feature_pairs:
#     fig, ax = plt.subplots(figsize=(10, 6))
#     PartialDependenceDisplay.from_estimator(
#         RF_model, X_train, [pair],  # Pass each pair as a tuple
#         grid_resolution=50, ax=ax
#     )
#     plt.suptitle(f'2D Partial Dependence Plot for Features {pair[0]} and {pair[1]}', 
#                  fontsize=16, fontweight='bold')
#     plt.show()


# #%% partial dependency of 3 features

# # n_features_to_plot = 3

# # # Generate combinations of features
# # feature_combinations = list(combinations(import_features, n_features_to_plot))

# # # Loop through each combination of features
# # for combo in feature_combinations:
# #     feature_1, feature_2, fixed_feature = combo  # Take 2D plot with fixed 3rd feature
    
# #     # Fix the 3rd feature value to its mean for simplicity
# #     fixed_value = X_train[fixed_feature].mean()
# #     X_train_temp = X_train.copy()
# #     X_train_temp[fixed_feature] = fixed_value  # Fix one feature
    
# #     # Generate 2D PDP for feature_1 and feature_2
# #     fig, ax = plt.subplots(figsize=(10, 6))
# #     PartialDependenceDisplay.from_estimator(
# #         RF_model, X_train_temp, [(feature_1, feature_2)], grid_resolution=50, ax=ax
# #     )
# #     plt.suptitle(f'2D PDP for Features {feature_1} & {feature_2} (Fixed {fixed_feature}={fixed_value:.2f})', 
# #                  fontsize=14, fontweight='bold')
# #   plt.show()

# #%% Initialize SHAP explainer

# # Create SHAP Explainer
# explainer = shap.Explainer(RF_model, X_train)
# shap_values = explainer(X_train)

# # Loop through all pairs of important features
# for feature_pair in combinations(import_features, 2):  # Pairwise combinations
#     feature_1, feature_2 = feature_pair
    
#     # SHAP Scatter Plot for the pair of features
#     plt.figure(figsize=(8, 6))
#     shap.plots.scatter(shap_values[:, feature_1], color=shap_values[:, feature_2])
#     plt.title(f'SHAP Interaction: {feature_1} and {feature_2}', fontsize=14, fontweight='bold')
#     plt.show()

#%% Feature Information
import_features = [8, 13, 18, 19, 33, 38, 39, 42, 59]
import_feature_names = ['Residue number 3 Electronic Nature', 'Residue number 5 Size', 'Residue number 7 Hydrophilicity', 
                        'Residue number 7 Size', 'Residue number 12 Hydrophilicity', 'Residue number 13 Electronic Nature',
                        'Residue number 14 Hydrophilicity', 'Residue number 15 Hydrophilicity', 'Residue number 20 Electronic Nature']
import_feature_zscale_no = ['z3', 'z2', 'z1', 'z2', 'z1', 'z3', 'z1', 'z1', 'z3']

# #%% 1D Partial Dependence Plots (Seaborn Visualization)

# # Loop through all important features
# for i, feature_idx in enumerate(import_features):
#     feature_name = import_feature_names[i]
#     z_scale_val = import_feature_zscale_no[i]

#     # Compute partial dependence values
#     pd_result = partial_dependence(RF_model, X_train, [feature_idx], grid_resolution=100)
    
#     # Extract grid and partial dependence values
#     pd_values = pd_result.average[0]  # Average partial dependence values
#     pd_grid = pd_result.grid_values[0]  # Grid points for the feature

#     # Plot using Seaborn
#     plt.figure(figsize=(60,60))
#     sns.lineplot(x=pd_grid, y=pd_values, color='blue', lw=20)
#     plt.title(f'Partial Dependence: {feature_name}', fontsize=150, fontweight='bold', fontname = 'arial')
#     plt.xlabel(z_scale_val, fontsize=150, fontname = 'arial', fontweight='bold')
#     plt.ylabel('Partial Dependence Value', fontsize=150, fontname = 'arial', fontweight='bold')
#     plt.xticks(fontsize = 150, fontname = 'Arial', fontweight='bold')
#     plt.yticks(fontsize = 150, fontname = 'Arial', fontweight='bold')
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(f'Partial_Dependency_Plot_of_{import_feature_names[i]}.png', dpi = 500)
#     plt.show()

# #%% 2D Partial Dependence Plots (Seaborn Heatmaps)
# feature_pairs = list(combinations(import_features, 2))

# for pair in feature_pairs:
#     idx1, idx2 = pair
#     name1 = import_feature_names[import_features.index(idx1)]
#     name2 = import_feature_names[import_features.index(idx2)]
    
#     pd_results = partial_dependence(RF_model, X_train, features=[(idx1, idx2)], grid_resolution=20)
#     pd_values = pd_results['average'].reshape(20, 20)
#     pd_grid1 = pd_results['grid_values'][0]
#     pd_grid2 = pd_results['grid_values'][1]

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(pd_values, xticklabels=np.round(pd_grid2, 2), yticklabels=np.round(pd_grid1, 2), cmap="viridis")
#     plt.title(f'2D Partial Dependence: {name1} vs {name2}', fontsize=14, fontweight='bold')
#     plt.xlabel(name2, fontsize=12)
#     plt.ylabel(name1, fontsize=12)
#     plt.tight_layout()
#     plt.show()
#%% SHAP 1

# #%% SHAP Value Plots for Feature Pairs
# explainer = shap.Explainer(RF_model, X_train)
# shap_values = explainer(X_train)

# # Extract SHAP values as a numpy array
# shap_values_array = shap_values.values  # SHAP values for all features

# # Loop through pairs of important features
# for idx1 in range(9): 
#     for idx2 in range(9):
#         plt.figure(figsize=(8, 6))
#         scatter = plt.scatter(
#             shap_values_array[:, import_features[idx1]],  # SHAP values for feature 1
#             shap_values_array[:, import_features[idx2]],  # SHAP values for feature 2
#             c=shap_values_array[:, import_features[idx2]],  # Color based on SHAP values for feature 2
#             cmap='viridis', edgecolor='k', s=60
#             )
#         plt.colorbar(scatter, label=f'SHAP interaction Values of {import_feature_names[idx1]} and {import_feature_names[idx2]}')  # Corrected colorbar label
    
#         # Set labels with corrected feature names
#         plt.xlabel(f'SHAP Values for {import_feature_names[idx1]}', fontsize=12, fontname = 'Arial', fontweight='bold')
#         plt.ylabel(f'SHAP Values for {import_feature_names[idx2]}', fontsize=12, fontname = 'Arial', fontweight='bold')
#         plt.title(f'SHAP Interaction: {import_feature_names[idx1]} and {import_feature_names[idx2]}',
#                   fontsize=14, fontweight='bold', fontname = 'Arial')
    
#         plt.grid(True)  # Enable grid
#         plt.tight_layout()  # Improve spacing
#         plt.show()

# #%% SHAP 2

# # Create SHAP explainer (for a tree-based model, like RandomForest)
# explainer = shap.TreeExplainer(RF_model, X_train)

# # Calculate SHAP values (including interaction values)
# shap_values = explainer.shap_values(X_train)

# # For interaction values, you would use the interaction_values attribute
# interaction_values = shap_values[0].interaction_values  # For the first class in classification or regression

# # Loop through pairs of important features
# for idx1 in range(9):
#     for idx2 in range(9):
#         # Get the interaction values for the pair of features idx1 and idx2
#         # Interaction values for feature pair idx1, idx2 are stored in shap_values[0].interaction_values
#         #interaction_values = shap_values[0].interaction_values[:, import_features[idx1], import_features[idx2]]
        
#         # Calculate the maximum interaction value
#         max_interaction_value = np.max(np.abs(interaction_values))  # Maximum interaction value between features idx1 and idx2
        
#         # Only plot if the maximum interaction value is greater than 0.5
#         if max_interaction_value > 0.5:
#             # Create a dataframe for seaborn plot
#             data = {
#                 f'SHAP Values for {import_feature_names[idx1]}': shap_values[0].values[:, import_features[idx1]],
#                 f'SHAP Values for {import_feature_names[idx2]}': shap_values[0].values[:, import_features[idx2]],
#                 'Interaction SHAP Values': interaction_values  # Interaction SHAP values for coloring
#             }

#             df = pd.DataFrame(data)
            
#             # Use Seaborn to create the scatter plot
#             plt.figure(figsize=(8, 6))
#             scatter = sns.scatterplot(
#                 x=f'SHAP Values for {import_feature_names[idx1]}',
#                 y=f'SHAP Values for {import_feature_names[idx2]}',
#                 hue='Interaction SHAP Values',  # Color the points based on SHAP interaction values
#                 palette='viridis',  # Use the 'viridis' colormap
#                 data=df,
#                 s=60,  # Marker size
#                 edgecolor='k',  # Black edge color for points
#                 legend=None  # Remove legend
#             )
            
#             # Add colorbar with Arial font
#             norm = plt.Normalize(df['Interaction SHAP Values'].min(), df['Interaction SHAP Values'].max())  # Normalize the color scale
#             sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
#             sm.set_array([])  # Empty array for the colorbar
#             cbar = plt.colorbar(sm, ax=plt.gca())  # Add colorbar
#             cbar.ax.tick_params(labelsize=12)
#             # cbar.ax.set_xlabel(f'SHAP interaction', fontsize=12, fontname='Arial')

#             # Set labels and title with Arial font
#             plt.xlabel(f'SHAP Values for {import_feature_names[idx1]}', fontsize=12, fontname='Arial', fontweight='bold')
#             plt.ylabel(f'SHAP Values for {import_feature_names[idx2]}', fontsize=12, fontname='Arial', fontweight='bold')
#             plt.title(f'SHAP Interaction: {import_feature_names[idx1]} and {import_feature_names[idx2]}',
#                       fontsize=14, fontweight='bold', fontname='Arial')

#             plt.grid(True)  # Enable grid
#             plt.tight_layout()  # Improve spacing

#             # Save the plot with high DPI
#             plt.savefig(f'shap_interaction_{import_feature_names[idx1]}_{import_feature_names[idx2]}.png', dpi=500)
#             plt.show()
            
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
        if max_interaction_value > 1:
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