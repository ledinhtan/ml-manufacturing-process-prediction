import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from model_trainer import ModelTrainer

def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(drop='first', sparse_output=False), ['tape_id'])
        ],
        remainder='passthrough'
    )

def main():
    # Create the new folder--trained_models
    os.makedirs("trained_models", exist_ok=True)

    # Load training & testing sets
    train_df = pd.read_csv("./data/processed/train_data.csv")
    test_df = pd.read_csv("./data/processed/test_data.csv")

    # ================================================================ Pipelines for all models ================================================================
    pipelines = {
        "green_density": Pipeline([
            ("model", ExtraTreesRegressor(bootstrap=False, max_features=0.95, min_samples_leaf=1, min_samples_split=3, n_estimators=100, random_state=42)) 
        ]),
        "green_thickness": Pipeline([
            ("model", ExtraTreesRegressor(bootstrap=False, max_features=0.95, min_samples_leaf=1, min_samples_split=3, n_estimators=100, random_state=42))    
        ]),
        "sintered_density": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", ExtraTreesRegressor(bootstrap=False, max_features=0.95, min_samples_leaf=1, min_samples_split=3, n_estimators=100, random_state=42))
        ]),
        "sintered_thickness": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", ExtraTreesRegressor(bootstrap=False, max_features=0.95, min_samples_leaf=1, min_samples_split=3, n_estimators=100, random_state=42))
        ]),
        "reduced_density": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", RidgeCV())
        ]),
        "reduced_thickness": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", RidgeCV())
        ]),
    }

    # ================================================================ Task-specific configuration ================================================================
    tasks = {
        "green_density": {
            "input_columns": ["temperature", "doctor_blade", "casting_speed", "humidity", "volume_flow_rate"],
            "target_column": "green_density",
            "plot": {
                "title": "Green Density",
                "xlabel": "Target Values [g/cm³]",
                "ylabel": "Predicted Values [g/cm³]",
                "color": "dodgerblue"
            }
        },
        "green_thickness": {
            "input_columns": ["temperature", "doctor_blade", "casting_speed", "humidity", "volume_flow_rate"],
            "target_column": "green_thickness",
            "plot": {
                "title": "Green Thickness",
                "xlabel": "Target Values [µm]",
                "ylabel": "Predicted Values [µm]",
                "color": "skyblue"
            }
        },
        "sintered_density": {
            "input_columns": ["tape_id", "green_thickness", "green_density"],
            "target_column": "sintering_density",
            "plot": {
                "title": "Sintered Density",
                "xlabel": "Target Values [g/cm³]",
                "ylabel": "Predicted Values [g/cm³]",
                "color": "darkturquoise"
            }
        },
        "sintered_thickness": {
            "input_columns": ["tape_id", "green_thickness", "green_density"],
            "target_column": "sintering_thickness",
            "plot": {
                "title": "Sintered Thickness",
                "xlabel": "Target Values [µm]",
                "ylabel": "Predicted Values [µm]",
                "color": "deepskyblue"
            }
        },
        "reduced_density": {
            "input_columns": ["tape_id", "sintering_thickness", "sintering_density"],
            "target_column": "reducing_density",
            "plot": {
                "title": "Reduced Density",
                "xlabel": "Target Values [g/cm³]",
                "ylabel": "Predicted Values [g/cm³]",
                "color": "royalblue"
            }
        },
        "reduced_thickness": {
            "input_columns": ["tape_id", "sintering_thickness", "sintering_density"],
            "target_column": "reducing_thickness",
            "plot": {
                "title": "Reduced Thickness",
                "xlabel": "Target Values [µm]",
                "ylabel": "Predicted Values [µm]",
                "color": "deeppink"
            }
        }
    }

    # ================================================================ Train all models ================================================================
    for name, cfg in tasks.items():
        print(f"\n Evaluation on the testing set for {name}") 
        trainer = ModelTrainer(
            pipeline=pipelines[name],
            train_set=train_df,
            test_set=test_df,
            target_column=cfg["target_column"],
            input_columns=cfg["input_columns"],
            plot_config=cfg["plot"]
        )
        trainer.train_and_evaluate()
        trainer.save_model(f"trained_models/{name}_model.pkl")

if __name__ == "__main__":
    main()
