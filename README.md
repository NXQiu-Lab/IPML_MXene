# Machine Learning Prediction of Hydrogen Adsorption Free Energy

This repository contains the feature-selection and machine-learning
workflow used for predicting hydrogen adsorption free energy (ΔG_H).

## Target property

The target property is:

`Delta_G_eV`

## Feature-selection workflow

Feature selection consists of two steps.

### 1. Pearson correlation filtering

### 2. RFECV

The descriptors remaining after Pearson filtering are further selected
using Recursive Feature Elimination with Cross-Validation (RFECV).

RFECV settings:

- Estimator: Random Forest Regressor
- Number of estimators: 100
- Maximum tree depth: 10
- Inner cross-validation: 10-fold
- Scoring metric: R²
- Minimum number of selected descriptors: 3
- Random state: 100

## Machine-learning models

Seven regression models are evaluated:

- Random Forest Regression (RFR)
- Gradient Boosting Regression (GBR)
- AdaBoost Regression (ABR)
- K-Nearest Neighbors (KNN)
- Kernel Ridge Regression (KRR)
- Partial Least Squares Regression (PLS)
- Ridge Regression

## Model evaluation

Model performance is evaluated using outer five-fold cross-validation:

`KFold(n_splits=5, shuffle=True, random_state=100)`

The reported metrics are:

- R²
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

Pearson filtering and RFECV are performed using the training samples
of each outer fold only. Validation samples are not used for feature
selection or feature scaling.

