import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_and_preprocess_data, train_and_log_model_with_mlflow

# Use raw strings or double backslashes for paths
DATASET_PATH = os.path.join('weather-forecast_csv', 'weatherHistory.csv')
# DATASET_PATH = r"weather-forecast_csv\weatherHistory.csv"

@pytest.fixture
def sample_data():
    """Fixture to provide a random sample of data from the dataset."""
    df = pd.read_csv(DATASET_PATH)
    sample = df.sample(n=10, random_state=1)  # Sample 10 random rows
    return sample

def test_load_and_preprocess_data(sample_data):
    """Test the data loading and preprocessing function."""
    features, labels = load_and_preprocess_data(DATASET_PATH)
    
    assert features.shape[1] == 7, "Features should have 7 columns after preprocessing (one column removed by drop_first in get_dummies)."
    assert len(labels) == len(features), "Labels length should match features length."
    assert not features.isnull().values.any(), "Features should not contain NaN values."

def test_train_model(sample_data):
    """Test the model training function."""
    features, labels = load_and_preprocess_data(DATASET_PATH)
    model = train_and_log_model_with_mlflow(features, labels)

    assert model is not None, "Model should not be None."
    assert hasattr(model, "predict"), "Model should have a predict method."
    assert hasattr(model, "fit"), "Model should have a fit method."
