# Machine Learning Prediction of Formation Energy

This repository contains the feature-selection and machine-learning
workflow used for the prediction of formation energy.

## Target property

The target property is:

`Formation_Energy_eV_per_atom`

## Workflow

The machine-learning workflow consists of:

1. Data preprocessing
2. 3-sigma target outlier filtering
3. Pearson correlation-based descriptor selection
4. Descriptor standardization
5. Regression model training
6. Five-fold cross-validation

## Feature selection

Highly correlated descriptors are identified using the Pearson
correlation coefficient.

For each pair satisfying:

`|r| > 0.80`

the descriptor showing the weaker absolute Pearson correlation with
the target property is removed.

During cross-validation, feature selection is performed independently
using each training fold to prevent validation data from being used
during descriptor selection.

## Machine-learning models

The following regression algorithms are evaluated:

- Random Forest Regression (RFR)
- Gradient Boosting Regression (GBR)
- AdaBoost Regression (ABR)
- K-Nearest Neighbors (KNN)
- Kernel Ridge Regression (KRR)
- Partial Least Squares Regression (PLS)
- Ridge Regression
- Multilayer Perceptron (MLP)

## Model evaluation

Models are evaluated using five-fold cross-validation:

`KFold(n_splits=5, shuffle=True, random_state=100)`

The reported metrics are:

- R²
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

