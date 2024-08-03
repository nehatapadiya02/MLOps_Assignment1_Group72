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

def train_and_log_model_with_mlflow(X, y, model, model_name, params, output_csv_path):
    # Start a new MLflow run
    with mlflow.start_run():
        try: 
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Feature scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train the model
            model.set_params(**params)
            model.fit(X_train, y_train)

            # # Train a simple linear regression model
            # model = LinearRegression()
            # model.fit(X_train, y_train)

            # Predict and Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f'{model_name} Mean Squared Error: {mse}')

            # Log parameters and metrics
            mlflow.log_param("model", model_name)
            for param, value in params.items():
                mlflow.log_param(param, value)
            mlflow.log_metric("mean_squared_error", mse)

             # Save predictions to CSV file
            if output_csv_path:
                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                results_df.to_csv(output_csv_path, index=False)

                # Log the CSV file as an artifact
                mlflow.log_artifact(output_csv_path)

            # Log the model and scaler
            mlflow.sklearn.log_model(model, "model")
            mlflow.sklearn.log_model(scaler, "scaler")

            # Save the model and the scaler for deployment
            joblib.dump(model, 'model.joblib')
            joblib.dump(scaler, 'scaler.joblib')

            print(f"{model_name} trained and saved successfully!")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally log the error as a MLflow tag
            mlflow.log_param("error", str(e))
            # Explicitly end the MLflow run with a failed status
            mlflow.end_run(status="FAILED")
            raise

        finally:
            # Print the output CSV file if it exists
            if output_csv_path and os.path.exists(output_csv_path):
                print("Contents of the output CSV file:")
                output_df = pd.read_csv(output_csv_path)
                print(output_df.head())
            # Ensure that the MLflow run is ended properly in case of exceptions
            mlflow.end_run()

if __name__ == "__main__":
    DATASET_PATH = os.path.join('weather-forecast_csv', 'weatherHistory.csv')
    # DATASET_PATH = r'weather-forecast_csv\weatherHistory.csv'
    X, y = load_and_preprocess_data(DATASET_PATH)

    # Define different models and their parameters
    models = [
        (RandomForestRegressor(), "RandomForestRegressor", {"n_estimators": 100, "max_depth": 10}),
        (GradientBoostingRegressor(), "GradientBoostingRegressor", {"n_estimators": 100, "learning_rate": 0.1}),
        (LinearRegression(), "LinearRegression", {})
    ]

    # Train and log each model
    for model, model_name, params in models:
        # Define a unique output CSV path for each model
        output_csv_path = f"{model_name}_predictions.csv"
        train_and_log_model_with_mlflow(X, y, model, model_name, params, output_csv_path)
