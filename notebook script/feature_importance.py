# -*- coding: utf-8 -*-
"""Capstone 1_ML_ZoomCamp_2024.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1z2ujDKVscio6n41CdP-CTAaOQkUu8IjB

### CAPSTONE PROJECT 1

### LIBRARIES :
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from sklearn.inspection import permutation_importance

"""### READ THE DATASET :"""

df = pd.read_csv('space_missions_dataset.csv')
df.head()

df.tail()

"""### FEATURE IMPORTANCE"""

def calculate_feature_importances(X, y):
    model = LinearRegression()
    model.fit(X, y)
    importances = pd.Series(model.coef_, index=X.columns)
    print("\nFeature Importances (Linear Regression):")
    print(importances)
    return importances

X = df[['Distance from Earth (light-years)', 'Mission Duration (years)',
        'Mission Cost (billion USD)', 'Scientific Yield (points)', 'Crew Size']]
y = df['Mission Success (%)']

feature_importances = calculate_feature_importances(X, y)

sorted_importances = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(sorted_importances.index, sorted_importances.values, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances (Linear Regression)')
plt.gca().invert_yaxis()
plt.show()

"""### FEATURE IMPORTANCE SCORES FROM MODEL AND VIA RFE"""

def calculate_rfe_importances(X, y):
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=1, verbose=0)
    rfe.fit(X, y)
    importances = pd.DataFrame({
        'Feature': X.columns,
        'RFE Importance': 1 / rfe.ranking_
    }).sort_values(by='RFE Importance', ascending=False)
    return importances

rfe_importances = calculate_rfe_importances(X, y)
print(rfe_importances)

def ranking(ranks, names, order=1):
    return sorted(zip(ranks, names), reverse=order == -1)

lr = LinearRegression()

rfe = RFE(estimator=lr, n_features_to_select=1, verbose=3)
rfe.fit(X, y)

ranks = {}
ranks["RFE_LR"] = ranking(list(map(float, rfe.ranking_)), X.columns, order=-1)

print("\nRFE Rankings:")
for rank, feature in ranks["RFE_LR"]:
    print(f"Feature: {feature}, Rank: {rank}")

"""### PERMUTABLE FEATURE IMPORTANCE"""

def get_permutation_importance(X, y, model):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
    result_df = pd.DataFrame({
        'Feature': X.columns,
        'Permutation Importance (Mean)': result.importances_mean,
        'Permutation Importance (Std Dev)': result.importances_std
    })
    return result_df

lr.fit(X, y)
permutate_df = get_permutation_importance(X, y, lr)

sorted_permutate_df = permutate_df.sort_values('Permutation Importance (Mean)', ascending=False)
print(sorted_permutate_df[['Feature', 'Permutation Importance (Mean)']].head(20))

"""### DROP-COLUMN IMPORTANCE"""

def drop_col_feat_imp(model, X, y):
    model_clone = clone(model)
    model_clone.fit(X, y)
    benchmark_score = model_clone.score(X, y)

    importances = []
    for col in X.columns:
        model_clone = clone(model)
        model_clone.fit(X.drop(col, axis=1), y)
        drop_col_score = model_clone.score(X.drop(col, axis=1), y)
        importance = (benchmark_score - drop_col_score) / benchmark_score
        importances.append(importance)

    importances_df = pd.DataFrame({'Feature': X.columns, 'Drop-Column Importance': importances})
    return importances_df.sort_values(by='Drop-Column Importance', ascending=False)

model = LinearRegression()

drop_col_impt_df = drop_col_feat_imp(model, X, y)

print(drop_col_impt_df)

sorted_drop_col_impt_df = drop_col_impt_df.sort_values('Drop-Column Importance', ascending=False)
print(sorted_drop_col_impt_df[['Feature', 'Drop-Column Importance']].head(20))

importances = drop_col_impt_df.set_index('Feature')['Drop-Column Importance']

importances = importances / importances.max() * 100

def plot_feature_importances(importances):
    plt.figure(figsize=(10, 6))
    ax = importances.sort_values().plot(kind='barh', color='skyblue')
    for i, v in enumerate(importances.sort_values()):
        plt.text(v + 1, i, f"{v:.2f}%", va='center')
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")
    plt.title("Feature Importances (Normalized)")
    plt.gca().invert_yaxis()
    plt.xlim(left=0, right=100)
    plt.show()

plot_feature_importances(importances)

"""### MERGING FEATURE IMPORTANCE METRICS INTO A SINGLE RESULT DATAFRAME"""

from tabulate import tabulate

all_ranks = pd.merge(
    drop_col_impt_df[['Feature', 'Drop-Column Importance']],
    permutate_df[['Feature', 'Permutation Importance (Mean)']],
    on='Feature'
)

all_ranks = pd.merge(
    all_ranks,
    rfe_importances[['Feature', 'RFE Importance']],
    on='Feature'
)

all_ranks['mean_feature_importance'] = (
    all_ranks['Drop-Column Importance'] +
    all_ranks['Permutation Importance (Mean)'] +
    all_ranks['RFE Importance']
) / 3

all_ranks = all_ranks.sort_values(by='mean_feature_importance', ascending=False).head(10)

print(tabulate(all_ranks.reset_index(drop=True), headers='keys', tablefmt='pretty'))

"""### CONCLUSION
Among the analyzed features, "Mission Cost (billion USD)" consistently stands out as the most influential factor in predicting mission success across all evaluation methods,
including Linear Regression coefficients, Recursive Feature Elimination (RFE), Permutation Importance, and Drop-Column Importance.
Other features, such as "Mission Duration (years)" and "Distance from Earth (light-years)," show varying levels of importance,
but their impact is significantly smaller compared to mission cost."""

















