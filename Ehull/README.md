# Machine Learning Prediction of Energy Above Hull

This repository contains the feature-selection and machine-learning
workflow used for the prediction of energy above the convex hull.

## Target property
The prediction target is:`E_above_hull_eV_per_atom`

## Workflow

The modeling workflow consists of:
1. Data preprocessing
2. 3-sigma target outlier filtering
3. Pearson correlation filtering
4. Recursive feature elimination with cross-validation (RFECV)
5. Machine-learning model training
6. Five-fold cross-validation

## Feature selection

### Pearson correlation filtering
For descriptor pairs satisfying
`|Pearson r| > 0.95`
the descriptor with the weaker absolute Pearson correlation with the
target is removed.

### RFECV

The remaining descriptors are further selected using RFECV with a
Random Forest regressor.

RFECV settings:

- Estimator: Random Forest Regressor
- Number of trees: 100
- Cross-validation: 10-fold
- Scoring metric: R²
- Minimum number of descriptors: 3
- Random state: 100

For model-performance evaluation, descriptor selection is performed
using the training portion of each outer cross-validation fold.

## Machine-learning models

Eight regression algorithms are evaluated:

- Random Forest Regression (RFR)
- Gradient Boosting Regression (GBR)
- AdaBoost Regression (ABR)
- K-Nearest Neighbors (KNN)
- Kernel Ridge Regression (KRR)
- Partial Least Squares Regression (PLS)
- Ridge Regression
- Multilayer Perceptron (MLP)

## Model evaluation

Model performance is evaluated by five-fold cross-validation.

Metrics:

- R²
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

