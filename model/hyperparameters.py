# model/hyperparameters.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import joblib
import os

def load_and_preprocess_data(data_path):
    data = pd.read_csv(data_path)
    X = data.drop(['Formatted Date', 'Summary', 'Daily Summary', 'Loud Cover', 'Temperature (C)'], axis=1)
    y = data['Temperature (C)']
    X = pd.get_dummies(X, columns=['Precip Type'], drop_first=True)
    return X, y

def tune_hyperparameters(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge()
    params = {'alpha': [0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Best model MSE: {mse}')

    joblib.dump(best_model, 'best_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    print("Best model saved successfully!")
    return best_model

if __name__ == "__main__":
    DATASET_PATH = os.path.join('weather-forecast_csv', 'weatherHistory.csv')
    X, y = load_and_preprocess_data(DATASET_PATH)
    best_model = tune_hyperparameters(X, y)