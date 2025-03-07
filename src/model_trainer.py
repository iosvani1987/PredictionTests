from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class ModelTrainer:
    def __init__(self, input_path: str, model_type: str = 'RF'):
        self.input_path = input_path
        self.model_type = model_type
        
        data_df = pd.read_csv(self.input_path)
        # Split the data into features and target
        X = data_df.drop('score', axis=1)
        y = data_df['score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None

    def train_models(self):
        if self.model_type == 'RF':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'LR':
            self.model = LogisticRegression(max_iter=10000)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        self.model.fit(self.X_train, self.y_train)


    def predict(self, X: pd.DataFrame):
        predictions = np.round(self.model.predict(X))
        return predictions

    def evaluate_models(self):
        # Calculate MSE
        mse = mean_squared_error(self.y_test, self.predict(self.X_test))
        return mse
