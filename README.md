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

Step-8 : Run models
    python model\model.py
    python model\hyperparameters.py

Step-9 : Run models
    pytest tests\test_model.py
    pytest tests\test_hyperparameters.py

Step-10 : Run CI/CD pipeline locally
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    irm get.scoop.sh | iex
    scoop install act
    act --version
    act

Step-11 : Build and Run Docker Container
  Build Docker image
    docker build -t weather-forecast-app .
  Run Docker Container
    docker run -d -p 5000:5000 weather-forecast-app
  Test Flask Application
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
    
    act -W "M1 MLOps Foundation/.github/workflows/ci-cd-pipeline.yml"
