import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
import mlflow
import mlflow.sklearn

def load_and_preprocess_data(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Data Preprocessing
    # Assuming 'Temperature (C)' is the target variable to predict
    X = data.drop(['Formatted Date', 'Summary', 'Daily Summary', 'Loud Cover', 'Temperature (C)'], axis=1)
    y = data['Temperature (C)']

    # Convert categorical data to numeric, if necessary
    X = pd.get_dummies(X, columns=['Precip Type'], drop_first=True)

    return X, y

def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the model and the scaler for deployment
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    print("Model trained and saved successfully!")
    return model

if __name__ == "__main__":
    DATASET_PATH = os.path.join('weather-forecast_csv', 'weatherHistory.csv')
    # DATASET_PATH = r'weather-forecast_csv\weatherHistory.csv'
    X, y = load_and_preprocess_data(DATASET_PATH)
    model = train_model(X, y)
    # model = train_and_log_model_with_mlflow(X,y)

