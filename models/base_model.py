import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#pip install pandas numpy matplotlib scikit-learn

class RegressionModel:
    def __init__(self, df, target_column, model=None, test_size=0.2, random_state=42):
        self.df = df.copy()
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        self.model = model if model is not None else RandomForestRegressor(
            n_estimators=200,
            random_state=random_state
        )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", self.model)
        ])

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.cv_results = None
        self.metrics = None

    def split_data(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        if self.X_train is None:
            self.split_data()

        self.pipeline.fit(self.X_train, self.y_train)
        self.y_pred = self.pipeline.predict(self.X_test)

        self.metrics = {
            "MAE": mean_absolute_error(self.y_test, self.y_pred),
            "RMSE": mean_squared_error(self.y_test, self.y_pred) ** 0.5,
            "R2": r2_score(self.y_test, self.y_pred)
        }

        return self.metrics

    def cross_validation(self, cv=5):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        self.cv_results = cross_validate(
            self.pipeline,
            X,
            y,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2"
            },
            return_train_score=True
        )

        return {
            "CV MAE": -self.cv_results["test_mae"].mean(),
            "CV RMSE": -self.cv_results["test_rmse"].mean(),
            "CV R2": self.cv_results["test_r2"].mean()
        }

    def get_results(self):
        return {
            "model": self.model,
            "metrics": self.metrics,
            "cross_validation": self.cv_results
        }

    def predict(self, new_data):
        return self.pipeline.predict(new_data)

    def plot_predictions(self):
        plt.figure(figsize=(7, 5))
        plt.scatter(self.y_test, self.y_pred)
        plt.xlabel("Real values")
        plt.ylabel("Predicted values")
        plt.title("Real vs Predicted")
        plt.show()

    def plot_residuals(self):
        residuals = self.y_test - self.y_pred

        plt.figure(figsize=(7, 5))
        plt.scatter(self.y_pred, residuals)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()

    def plot_error_distribution(self):
        residuals = self.y_test - self.y_pred

        plt.figure(figsize=(7, 5))
        plt.hist(residuals, bins=30)
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.title("Error Distribution")
        plt.show()
