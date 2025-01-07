# -*- coding: utf-8 -*-
"""Capstone 1_ML_ZoomCamp_2024.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qythwp1QKVRTGyEDkljQZVxjO6Zmk0sN

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

"""### TRAIN/VALIDATION/TEST SPLIT"""

df_temp, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_temp, test_size=0.25, random_state=1)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

target_train = df_train['Mission Success (%)'].values
target_validation = df_val['Mission Success (%)'].values
target_test = df_test['Mission Success (%)'].values

df_train.drop(columns=['Mission Success (%)'], inplace=True)
df_val.drop(columns=['Mission Success (%)'], inplace=True)
df_test.drop(columns=['Mission Success (%)'], inplace=True)

print(f"Training set shape: {df_train.shape}, Target shape: {target_train.shape}")
print(f"Validation set shape: {df_val.shape}, Target shape: {target_validation.shape}")
print(f"Test set shape: {df_test.shape}, Target shape: {target_test.shape}")

"""### DICTVECTORIZER"""

vectorizer = DictVectorizer(sparse=True)

train_records = df_train.to_dict(orient='records')
X_train_vectorized = vectorizer.fit_transform(train_records)

validation_records = df_val.to_dict(orient='records')
X_validation_vectorized = vectorizer.transform(validation_records)

test_records = df_test.to_dict(orient='records')
X_test_vectorized = vectorizer.transform(test_records)

print(f"Shape of vectorized training data: {X_train_vectorized.shape}")
print(f"Shape of vectorized validation data: {X_validation_vectorized.shape}")
print(f"Shape of vectorized test data: {X_test_vectorized.shape}")

"""### DIMENSIONALITY REDUCTION AND MODEL TRAINING WITHOUT PARAMETER TUNING (PCA)

## Linear regression
"""

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_vectorized.toarray())
X_validation_pca = pca.transform(X_validation_vectorized.toarray())
X_test_pca = pca.transform(X_test_vectorized.toarray())

linear_model = LinearRegression()
linear_model.fit(X_train_pca, target_train)


"""### CONCLUSION
The linear regression model with PCA demonstrates consistent performance across training, validation, and test sets. The R² values (~0.71-0.75) indicate that the model explains a significant portion of the variance, with moderate prediction errors (RMSE ~4.7-5.0, MAE ~3.8-4.0).
While effective, more advanced models or optimization can improve performance.

### RIDGE REGRESSION WITH GRID SEARCH FOR HYPERPARAMETER TUNING
"""

ridge_param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_param_grid, cv=5, scoring='r2')
ridge_grid.fit(X_train_vectorized, target_train)

print(f"Best Ridge Parameters: {ridge_grid.best_params_}")

best_ridge_model = ridge_grid.best_estimator_


"""### CONCLUSION
The Ridge Regression model with the best parameter (alpha = 100.0) demonstrates balanced performance across all datasets. It achieves an R² of 0.710 on the training set, 0.753 on the validation set, and 0.725 on the test set, indicating good generalization with consistent prediction accuracy.
While the model performs well, further optimization or advanced modeling may enhance results.

### OPTIMIZATION OF LASSO REGRESSION HYPERPARAMETERS USING GRIDSEARCHCV
"""

lasso_param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
lasso_grid = GridSearchCV(Lasso(), lasso_param_grid, cv=5, scoring='r2')
lasso_grid.fit(X_train_vectorized, target_train)

print(f"Best Lasso Parameters: {lasso_grid.best_params_}")

best_lasso_model = lasso_grid.best_estimator_

"""### CONCLUSION
The Lasso regression model with the optimized alpha value of 10.0 demonstrates balanced performance across training, validation, and test sets. While the training R² indicates a moderate fit (0.701), the validation and test R² values (0.756 and 0.725) highlight the model's ability to generalize well to unseen data.
This suggests that the chosen alpha effectively controls overfitting while maintaining prediction accuracy.

### DECISION TREE REGRESSOR
"""

dt_model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt_model.fit(X_train_vectorized, target_train)

"""### CONCLUSION
The Decision Tree model demonstrates strong performance on the training set with an R² of 0.911, indicating a good fit to the data.
However, the decrease in R² for the validation (0.758) and test (0.798) sets highlights a slight overfitting tendency.
Despite this, the model maintains reasonable generalization capabilities across unseen data, with acceptable RMSE and MAE values.
Further optimization or regularization could improve the balance between training and validation performance.

### XGB REGRESSOR
"""

xgb_model = XGBRegressor(
    n_estimators=20,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
xgb_model.fit(X_train_vectorized, target_train)

"""### CONCLUSION
The XGBoost model demonstrates strong predictive performance with an R² of 0.910 on the training set, indicating a well-fitted model.
However, the validation (R² = 0.747) and test (R² = 0.803) results suggest some generalization challenges, likely due to mild overfitting.
Further tuning regularization parameters or adjustments to the training process could enhance model performance on unseen data.

### RANDOM FOREST REGRESSOR
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_vectorized, target_train)

"""### CONCLUSION
The Random Forest Regressor demonstrates excellent performance with an R² of 0.973 on the training set, indicating a strong fit. The validation and test set results, with R² values of 0.802 and RMSE values of 4.443 and 3.231, respectively, highlight the model's ability to generalize well to unseen data.
This balance suggests that the model effectively captures patterns in the data while avoiding overfitting.

### GRADIENT BOOSTING REGRESSOR
"""

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_vectorized, target_train)

"""### CONCLUSION
The Gradient Boosting Regressor demonstrates strong performance, achieving high accuracy across all datasets. The model fits the training data well (R²: 0.960) and generalizes effectively to the validation (R²: 0.813) and test sets (R²: 0.879).
This indicates the model captures complex patterns without significant overfitting."""

model_predictions = {
    "Linear Regression": linear_model.predict(X_validation_pca),
    "Ridge Regression": best_ridge_model.predict(X_validation_vectorized),
    "Lasso Regression": best_lasso_model.predict(X_validation_vectorized),
    "Decision Tree": dt_model.predict(X_validation_vectorized),
    "Random Forest": rf_model.predict(X_validation_vectorized),
    "XGBoost": xgb_model.predict(X_validation_vectorized),
    "Gradient Boosting": gb_model.predict(X_validation_vectorized)
}

evaluation_results = {}
for model_name, predictions in model_predictions.items():
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(target_validation, predictions)),
        "MAE": mean_absolute_error(target_validation, predictions),
        "R2": r2_score(target_validation, predictions)
    }
    evaluation_results[model_name] = metrics

model_metrics = {
    "Model": list(evaluation_results.keys()),
    "RMSE": [metrics["RMSE"] for metrics in evaluation_results.values()],
    "R2 Score": [metrics["R2"] for metrics in evaluation_results.values()],
    "MAE": [metrics["MAE"] for metrics in evaluation_results.values()]
}
metrics_df = pd.DataFrame(model_metrics)

metrics_df

"""### MODEL PERFORMANCE COMPARSION
"""

plt.figure(figsize=(12, 8))
plt.barh(metrics_df['Model'], metrics_df['RMSE'], color='skyblue')
plt.xlabel('RMSE')
plt.ylabel('Model')
plt.title('Model Performance Comparison (Validation Set)')
plt.show()

print(f"Test set shape: {df_test.shape}, Target shape: {target_test.shape}")

"""### TRAINING MULTIPLE VARIATIONS OF NEURAL NETWORKS WITH TUNED PARAMETERS"""

input_dim = X_train_vectorized.shape[1]

def build_model(hp):
    model = Sequential()
    
    model.add(Dense(units=hp.Int('units_layer1', min_value=64, max_value=512, step=64), activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_layer1', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(Dense(units=hp.Int('units_layer2', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_layer2', min_value=0.0, max_value=0.5, step=0.1)))
    
    if hp.Choice('add_layer3', [True, False]):
        model.add(Dense(units=hp.Int('units_layer3', min_value=32, max_value=128, step=32), activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_layer3', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(Dense(1))
    
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd']),
        loss='mse',
        metrics=['mae']
    )
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_mae',
    max_epochs=50,
    factor=2,
    directory='my_tuner_dir',
    project_name='nn_hyperparameter_tuning'
)

early_stopping = EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True)

tuner.search(
    X_train_vectorized,
    target_train,
    epochs=5,
    validation_split=0.2,
    callbacks=[early_stopping]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    X_train_vectorized,
    target_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping]
)

best_model.save('best_nn_model.h5')
print("Best model saved as 'best_nn_model.h5'")

test_loss, test_mae = best_model.evaluate(X_test_vectorized, target_test)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

"""### CONCLUSION
The model's performance improved significantly throughout the training process, as indicated by the reduction in both training and validation loss.
Despite initial instability and high validation error in early epochs, the model achieved a Test Loss of 90.8738 and a Test MAE of 8.1958.
These results suggest that the model has learned to generalize reasonably well, though further tuning or adjustments to the architecture and hyperparameters might enhance stability and performance, particularly in the early epochs."""

def evaluate_nn_model(model, X, y):
    
    predictions = model.predict(X).flatten()
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE: {mae:.3f}')
    print(f'R²: {r2:.3f}\n')
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

nn_results = []

print('\nEvaluating Neural Network Model (Validation Set)')
val_metrics = evaluate_nn_model(best_model, X_validation_vectorized, target_validation)
nn_results.append({
    "Set": "Validation",
    "RMSE": val_metrics["RMSE"],
    "MAE": val_metrics["MAE"],
    "R2": val_metrics["R2"]
})

print('\nEvaluating Neural Network Model (Test Set)')
test_metrics = evaluate_nn_model(best_model, X_test_vectorized, target_test)
nn_results.append({
    "Set": "Test",
    "RMSE": test_metrics["RMSE"],
    "MAE": test_metrics["MAE"],
    "R2": test_metrics["R2"]
})

nn_results_df = pd.DataFrame(nn_results)
print(nn_results_df)


"""### NEURAL NETWORK PERFORMANCE COMPARSION
"""

plt.figure(figsize=(10, 6))
plt.bar(nn_results_df['Set'], nn_results_df['RMSE'], color='skyblue')
plt.xlabel('Dataset')
plt.ylabel('RMSE')
plt.title('Neural Network Performance Comparison')
plt.show()










