# tests/test_hyperparameters.py

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.hyperparameters import load_and_preprocess_data, tune_hyperparameters

class TestHyperparameters(unittest.TestCase):

    def setUp(self):
        self.data_path = 'weather-forecast_csv/weatherHistory.csv'

    def test_load_and_preprocess_data(self):
        X, y = load_and_preprocess_data(self.data_path)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)

    def test_tune_hyperparameters(self):
        X, y = load_and_preprocess_data(self.data_path)
        best_model = tune_hyperparameters(X, y)
        self.assertIsNotNone(best_model)

if __name__ == '__main__':
    unittest.main()