import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# ********************************** Set plot style **********************************
plt.style.use('seaborn-v0_8-white') 
matplotlib.use("Agg") 
matplotlib.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 25,
    "font.family": "Times New Roman"
})
plt.rc('font', family='Times New Roman', weight='bold')

# ********************************** Model trainer class **********************************
class ModelTrainer:
    def __init__(self, pipeline: Pipeline, train_set: pd.DataFrame, test_set: pd.DataFrame,
                 target_column: str, input_columns: list, plot_config: dict):
        self.pipeline = pipeline
        self.train_set = train_set
        self.test_set = test_set
        self.target_column = target_column
        self.input_columns = input_columns
        self.plot_config = plot_config
        self.model = None

    def extract_X_y(self, df: pd.DataFrame):
        """Extracting the data for model training"""
        X = df[self.input_columns]
        y = df[self.target_column]
        return X, y

    def train_and_evaluate(self):
        """Train models on the training set and evaluate model on the testing set"""
        X_train, y_train = self.extract_X_y(self.train_set)
        X_test, y_test = self.extract_X_y(self.test_set)

        self.model = self.pipeline.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        _r2_score = r2_score(y_test, y_pred) # R2
        mae = mean_absolute_error(y_test, y_pred) # MAE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE
        print(f"Model for {self.target_column}")
        print(f"R2 score for {self.target_column}: {_r2_score:.5f}")
        print(f"MAE score for {self.target_column}: {mae:.5f}")
        print(f"RMSE score for {self.target_column}: {rmse:.5f}")
        self.plot_predictions(y_test, y_pred)

    def save_model(self, path: str):
        """Save model"""
        if self.model:
            joblib.dump(self.model, path)
            print(f"Model saved to {path}")

    def plot_predictions(self, y_true, y_pred):
        """Plot the regression plots"""
        plt.figure(figsize=(8, 7))
        plt.scatter(y_true, y_pred, s=60,
                    color=self.plot_config["color"],
                    edgecolors='k', label="Test sample")
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val],
                 linestyle='--', color='k', label="Fitting line")

        plt.xlabel(self.plot_config["xlabel"])
        plt.ylabel(self.plot_config["ylabel"])
        # plt.title(self.plot_config["title"])
        plt.legend(frameon=True, framealpha=0.8)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', width=2, length=6, pad=8, labelsize=20)
        plt.tight_layout()
        filename = f"trained_models/{self.target_column}"
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf", dpi=600)
        plt.close()






