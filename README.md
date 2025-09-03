# Titanic - Kaggle competition - Machine Learning 

## Overview
This notebook aims to find the best model for predictions on the **Kaggle Titanic competion** data using  scikit‑learn. The workflow includes: data loading, data exploration and plotting, feature engineering, preprocessing, model evaluation, hyperparameter tuning, model fitting/evaluating and feature importance visualisation.

Data and info available at:
https://www.kaggle.com/competitions/titanic

## Data
This notebook loads two CSV‑files (Kaggle):
- `data/test.csv`
- `data/train.csv`

Features in original data:
`Age`, `Cabin`, `Embarked`, `Fare`, `Name`, `Parch`, `Pclass`, `Sex`, `SibSp`, `Survived`

Engineered features:
'FamilySize','No_family','HasCabin', 'Title'

## Model & Pipeline
Evaluated models: 
Logistic regression, Decision Tree, Random Forest, Gradient Boosting, SVC

Hyperparameter-tuning:
Executed for three best models: Logistic regression, Random Forest, Gradient Boosting

Evaluation
Used metric: roc_auc_score

## Requirements
`python>=3.9`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

## Output
The file generates:
-the AUC scores of the 5 evaluated models
-the AUC scores after hyperparameter tuning of the 3 best models

It trains the best model on the full train data and makes predictions on the test data. The predictions are saved as submission.csv 



