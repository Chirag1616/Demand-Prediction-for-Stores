# src/train_walmart.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
import joblib

MODEL_MAPPING = {
    "RandomForest": RandomForestRegressor,
    "ExtraTrees": ExtraTreesRegressor,
    "Ridge": Ridge,
    "Lasso": Lasso
}


def create_model(model_name, params):
    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Model '{model_name}' not supported. Choose from {list(MODEL_MAPPING.keys())}")
    return MODEL_MAPPING[model_name](**params)

def load_data(path="data/walmart.csv"):
    return pd.read_csv(path)

def train_and_log(data_path, n_estimators=100, max_depth=None):
    mlflow.set_experiment("walmart-sales")

    data = load_data(data_path)
    X = data.drop("Weekly_Sales", axis=1)
    y = data["Weekly_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds, squared=False)

        # Log params and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)

        # Save and log model
        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"✅ Run logged with RMSE: {rmse}")
