# tests/test_model.py

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import load_and_preprocess_data, train_model

class TestModel(unittest.TestCase):

    def setUp(self):
        self.data_path = 'weather-forecast_csv/weatherHistory.csv'

    def test_load_and_preprocess_data(self):
        X, y = load_and_preprocess_data(self.data_path)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)

    def test_train_model(self):
        X, y = load_and_preprocess_data(self.data_path)
        model = train_model(X, y)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()