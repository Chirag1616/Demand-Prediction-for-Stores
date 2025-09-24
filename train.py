
"""
train.py -- MLOps-ready training script for Walmart sales (from your notebook)

Usage example:
    python train.py --data-path data/walmart.csv --n-estimators 200 --max-depth 10 --output-dir outputs --drift-current data/new_batch.csv

Requirements:
    pip install mlflow scikit-learn pandas numpy joblib evidently

What it does:
    - Loads CSV (expects columns like: Store, Dept, Date, Weekly_Sales, Temperature, Fuel_Price, CPI, Unemployment, IsHoliday)
    - Feature engineering:
        * datetime parsing, Year/Week/Month
        * cyclical encodings for week/month
        * group lags and rolling mean per Store
    - Trains RandomForestRegressor
    - Logs params, metrics, model and artifacts to MLflow
    - Optionally runs Evidently DataDriftPreset between reference (train) and a provided current dataset
"""

import os
import argparse
import logging
from datetime import datetime
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso

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



try:
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except Exception:
    EVIDENTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def load_data(path):
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)
    # Try to parse Date if present
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception:
            logger.warning("Could not parse 'Date' column automatically.")
    return df

def feature_engineering(df, target_col="Weekly_Sales", lags=(1,2,3)):
    df = df.copy()
    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input data. Columns: {list(df.columns)}")

    # Parse date and create calendar features
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

        df = df.sort_values('Date')
        df['Year'] = df['Date'].dt.year
        df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
        df['Month'] = df['Date'].dt.month
    else:
        # create dummy time columns if no Date
        logger.warning("No 'Date' column found. Creating dummy Year/Week/Month = 0")
        df['Year'] = 0
        df['Week'] = 0
        df['Month'] = 0

    # Cyclical encodings
    df['week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Fill missing numeric columns with median (Temperature, Fuel_Price, CPI, Unemployment) if present
    num_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            if df[c].isna().any():
                med = df[c].median()
                df[c] = df[c].fillna(med)
                logger.info(f"Filled NA in {c} with median={med}")

    # Create interaction features seen in notebook
    if 'CPI' in df.columns and 'Fuel_Price' in df.columns:
        df['CPI_Fuel'] = df['CPI'] * df['Fuel_Price']
    if 'Temperature' in df.columns and 'Unemployment' in df.columns:
        df['Temp_Unemployment'] = df['Temperature'] * df['Unemployment']

    # Create lags and rolling mean per Store if Store exists, else global
    group_col = 'Store' if 'Store' in df.columns else None
    if group_col:
        for lag in lags:
            df[f'sales_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
        df['sales_roll_mean_3'] = df.groupby(group_col)[target_col].shift(1).rolling(window=3).mean().reset_index(level=0, drop=True)
    else:
        logger.warning("No 'Store' column found. Creating global lags/rolling.")
        for lag in lags:
            df[f'sales_lag_{lag}'] = df[target_col].shift(lag)
        df['sales_roll_mean_3'] = df[target_col].shift(1).rolling(window=3).mean()

    # IsHoliday to int if present
    if 'IsHoliday' in df.columns:
        df['IsHoliday'] = df['IsHoliday'].astype(int)

    # Fill remaining NaNs in lag/features by dropping rows with NaN in target or in important features
    features_after = ['sales_lag_1','sales_roll_mean_3']
    all_drop_cols = [target_col] + [c for c in features_after if c in df.columns]
    df = df.dropna(subset=all_drop_cols).reset_index(drop=True)
    logger.info(f"After creating lags, data shape: {df.shape}")
    return df

def prepare_features(df, target_col="Weekly_Sales"):
    df = df.copy()
    # Identify feature columns - take common numeric features + engineered ones
    candidate_feats = [
        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'CPI_Fuel', 'Temp_Unemployment',
        'Year', 'Week', 'Month',
        'week_sin', 'week_cos', 'month_sin', 'month_cos',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_roll_mean_3',
        'IsHoliday'
    ]
    features = [c for c in candidate_feats if c in df.columns]

    # Handle categorical Store/Dept by LabelEncoding (simple)
    cat_cols = []
    for c in ['Store','Dept']:
        if c in df.columns:
            # convert to string then label encode
            df[c] = df[c].astype(str)
            le = LabelEncoder()
            df[c+'_enc'] = le.fit_transform(df[c])
            features.append(c+'_enc')
            cat_cols.append(c+'_enc')

    X = df[features]
    y = df[target_col].astype(float)

    # final NaN handling
    if X.isna().any().any():
        logger.info("Filling remaining NA in features with medians.")
        X = X.fillna(X.median())

    return X, y, df

def train_and_log(data_path,
                  output_dir="outputs",
                  test_size=0.2,
                  random_state=42,
                  experiment_name="walmart-sales",
                  run_name=None,
                  mlflow_tracking_uri=None,
                  drift_current_path=None,
                  model_name="RandomForest",
                  model_params=None):
    model_params = model_params or {}

    safe_mkdir(output_dir)
    # Optionally set MLflow tracking URI
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(experiment_name)
    run_name = run_name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Load and prepare data
    df = load_data(data_path)
    df = feature_engineering(df, target_col="Weekly_Sales")
    X, y, df_proc = prepare_features(df, target_col="Weekly_Sales")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    with mlflow.start_run(run_name=run_name) as run:
        # Log params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("feature_cols", json.dumps(list(X.columns)))

        logger.info("Training RandomForestRegressor ...")
        model_params = model_params or {}
        model = create_model(model_name, model_params)
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))

        # Save model locally and log as artifact
        model_path = os.path.join(output_dir, "model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model_artifacts")

        # Log sklearn model for convenient loading
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Save sample predictions
        preds_df = X_test.copy()
        preds_df['y_true'] = y_test.values
        preds_df['y_pred'] = y_pred
        preds_sample_path = os.path.join(output_dir, "preds_sample.csv")
        preds_df.head(200).to_csv(preds_sample_path, index=False)
        mlflow.log_artifact(preds_sample_path, artifact_path="predictions")

        # Save processed dataframe sample for debugging
        df_sample_path = os.path.join(output_dir, "processed_sample.csv")
        df_proc.head(400).to_csv(df_sample_path, index=False)
        mlflow.log_artifact(df_sample_path, artifact_path="data_samples")

        # Optionally run Evidently drift check
        if drift_current_path:
            if not EVIDENTLY_AVAILABLE:
                logger.warning("Evidently not installed. Install with `pip install evidently` to enable drift checks.")
            else:
                logger.info("Running Evidently data drift report ...")
                try:
                    cur = pd.read_csv(drift_current_path)
                    # Ensure date parsing for current
                    if 'Date' in cur.columns:
                        cur['Date'] = pd.to_datetime(cur['Date'])
                    report = Report(metrics=[DataDriftPreset()])
                    report.run(reference_data=df_proc, current_data=cur)
                    drift_html = os.path.join(output_dir, "drift_report.html")
                    report.save_html(drift_html)
                    mlflow.log_artifact(drift_html, artifact_path="drift_reports")
                    logger.info(f"Drift report saved to {drift_html} and logged to MLflow.")
                except Exception as e:
                    logger.exception("Failed to produce drift report: %s", e)

        logger.info("Run complete. MLflow run id: %s", mlflow.active_run().info.run_id)
        print("✅ Training run logged to MLflow. Visit your MLflow UI to inspect.")

        return run.info.run_id 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to input CSV (training/reference data).")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to write outputs/artifacts.")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="walmart-sales")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--drift-current", type=str, default=None, help="Optional: path to current (new) CSV for data drift check by Evidently.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_id = train_and_log(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        drift_current_path=args.drift_current
    )
    logger.info("Finished. Run ID: %s", run_id)
