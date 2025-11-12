import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import mlflow
import mlflow.sklearn
import os
import glob

# --- STEP 0: Create required folders ---
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename=f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set MLflow to use local tracking
mlflow_tracking_path = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(f"file:///{mlflow_tracking_path.replace(os.sep, '/')}")

# Set experiment
experiment_name = "TimeSeries_Forecasting_Models"
mlflow.set_experiment(experiment_name)

try:
    logging.info("Training started")

    # --- STEP 1: Pick the latest data file ---
    all_files = glob.glob("data/Rawdata*.csv")
    if not all_files:
        raise FileNotFoundError("No data files found in 'data/' folder!")
    latest_file = max(all_files, key=os.path.getmtime)
    logging.info(f"Using data file: {latest_file}")
    print(f"Using data file: {latest_file}")

    # Load data
    data = pd.read_csv(latest_file)
    logging.info(f"Data loaded successfully, shape={data.shape}")

    # Preprocess date features
    data['OnRentDate'] = pd.to_datetime(data['OnRentDate'], errors='coerce')
    data = data.sort_values(by='OnRentDate')
    data['day'] = data['OnRentDate'].dt.day
    data['month'] = data['OnRentDate'].dt.month
    data['year'] = data['OnRentDate'].dt.year
    data['dayofweek'] = data['OnRentDate'].dt.dayofweek

    # Features and target
    X = data[['day', 'month', 'year', 'dayofweek']]
    y = data['Final_bookings']

    # Models
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    tscv = TimeSeriesSplit(n_splits=3)
    results = {}

    # Train each model
    for model_name, model in models.items():
        logging.info(f"Training model: {model_name}")
        mse_scores = []

        with mlflow.start_run(run_name=model_name):
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                mse_scores.append(mse)

            avg_mse = np.mean(mse_scores)
            results[model_name] = avg_mse

            # Log to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("avg_mse", avg_mse)

            # Save model with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_name}_model_{timestamp}.pkl"
            model_path = os.path.join("models", model_filename)
            joblib.dump(model, model_path)
            print(f"Saved model: {model_path}")
            logging.info(f"Saved model: {model_path}")

            # Log model as MLflow artifact
            mlflow.log_artifact(model_path)

    # --- Debug: List all saved models ---
    saved_models = os.listdir("models")
    logging.info(f"All saved models: {saved_models}")
    print("Models saved in 'models/' folder:", saved_models)

    if not saved_models:
        raise FileNotFoundError("No models were saved! GitHub Actions artifact upload will fail.")

    # Select best model
    best_model_name = min(results, key=results.get)
    logging.info(f"Best model: {best_model_name} with MSE = {results[best_model_name]:.4f}")
    print(f"Best model: {best_model_name} with MSE = {results[best_model_name]:.4f}")

except Exception as e:
    logging.exception(f"Error during training: {e}")
    print(f"Error during training: {e}")
    raise
