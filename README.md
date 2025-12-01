Machine Learning Models for BMP-2 Peptide Configuration Prediction
📘 Overview

This repository contains the code used in the research paper “Machine learning models for predicting configuration of modified knuckle epitope peptides of BMP-2 protein using mesoscale simulation data.” It includes preprocessing scripts, feature-scaling code, ML model definitions, and analysis scripts for explainability (PDP, SHAP), enabling users to reproduce and extend the results.

🗂 Repository Structure
/
├── Bayesian Optimization Random Forest.py      # ML model (Random Forest + hyperparameter tuning)  
├── PDP and SHAP EtE.py                         # Partial Dependence / SHAP analysis for model explainability  
├── SHAP plots Rg.py                            # Scripts to visualize SHAP results (e.g. radius of gyration, Rg)  
├── Input_sequences_z_scale.m                   # MATLAB(?) script for z-scaling / preprocessing sequences  
├── *.xlsx                                      # Data files (e.g. sequence sets, feature-scaled datasets)  
└── …                                          # Other auxiliary files for data, preprocessing, and analysis  

