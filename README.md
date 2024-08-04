# MLOps_Assignment1_Group72
BITS MTech Sem 3 MLOps Assignment-1 
Group 72

Step-1 : Create Virtual environment 
    python -m venv venv
    
Step-2 : Activate the environment
    venv\Scripts\activate {Windows} [OR]
    venv/bin/activate {Linux} [OR]
    .\venv\Scripts\Activate.ps1

Step-3 : If above command to activate environment doesnt work in windows
         Open powershell in run as administrator mode
         Run this command : Set-ExecutionPolicy Unrestricted -Force
         Try again with command given in Step-2

Step-4 : Install the requirements
    pip install -r requirements.txt
    
Step-5 : Run the application
    python main.py
    {If no error is thrown}

Step-6 : Run flake8 to lint your Python code
    flake8 main.py

Step-7 : Run tests
    pytest tests
    pytest -m "tests/test_weather_forecast"

Step-8 : MLflow to track experiments of project
    python main.py
    mlflow ui

Step-9 : Run models
    python model\model.py
    python model\hyperparameters.py

Step-10 : Run models
    pytest tests\test_model.py
    pytest tests\test_hyperparameters.py

Step-11 : Run CI/CD pipeline locally
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    irm get.scoop.sh | iex
    scoop install act
    act --version
    act -W ".github/workflows/ci-cd-pipeline.yml"

Step-12 : Build and Run Docker Container
  # Build Docker image
    docker build -t weather-forecast-app .
  # Run Docker Container
    docker run -d -p 5000:5000 weather-forecast-app
  # Test Flask Application
    Invoke-WebRequest -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '[
    {
        
        "Apparent Temperature (C)": 22.5,
        "Humidity": 0.75,
        "Wind Speed (km/h)": 10,
        "Wind Bearing (degrees)": 250,
        "Visibility (km)": 10.2,
        "Pressure (millibars)": 1015,
        "Precip Type_snow": 1
    }
  ]'

  Step-13 : DVC
  # Initialize Git and DVC
    git init
    dvc init
    
  # Add initial dataset
    dvc add weather-forecast_csv/weatherHistoryupdated.csv
    git add weather-forecast_csv/weatherHistoryupdated.csv.dvc .gitignore
    git commit -m "Add initial version of weatherHistoryupdated.csv"

  # Update dataset to second version
    echo "new data entry" >> weather-forecast_csv/weatherHistoryupdated.csv
    dvc add weather-forecast_csv/weatherHistoryupdated.csv
    git add weather-forecast_csv/weatherHistoryupdated.csv.dvc
    git commit -m "Update weatherHistoryupdated.csv with new data"

  # Update dataset to third version
    echo "another new data entry" >> weather-forecast_csv/weatherHistoryupdated.csv
    dvc add weather-forecast_csv/weatherHistoryupdated.csv
    git add weather-forecast_csv/weatherHistoryupdated.csv.dvc
    git commit -m "Add another update to weatherHistoryupdated.csv"

  # View dataset version history
    git log

  # Revert to a previous dataset version
    git checkout commit 7e21b9d83708512f7cde452196577ea810040507
    dvc checkout


  
    
    
