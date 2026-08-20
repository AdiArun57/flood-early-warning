# train_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate physics-grounded synthetic training baseline for extreme events
def create_hydrological_training_data(n_samples=5000):
    np.random.seed(42)
    elevation = np.random.uniform(1.0, 50.0, n_samples)
    rain_72h = np.random.uniform(0.0, 450.0, n_samples)
    peak_hourly_rain = rain_72h * np.random.uniform(0.1, 0.35, n_samples)
    soil_saturation = np.random.uniform(0.2, 0.95, n_samples)
    river_discharge = np.random.uniform(0.5, 45.0, n_samples)
    drainage_eff = np.random.uniform(0.1, 0.9, n_samples)
    
    # Physics formula: Infiltration excess + river backflow - natural gravity drainage
    runoff_load = (rain_72h * (1.0 - drainage_eff) * (1.0 + soil_saturation))
    river_surge = river_discharge * 0.08
    drain_potential = elevation * 0.05
    
    water_depth = np.maximum(0.0, (runoff_load * 0.015 + river_surge) - drain_potential + np.random.normal(0, 0.05, n_samples))
    
    return pd.DataFrame({
        "elevation_meters": elevation,
        "rain_72h_mm": rain_72h,
        "peak_hourly_rain_mm": peak_hourly_rain,
        "soil_saturation": soil_saturation,
        "river_discharge_m3s": river_discharge,
        "drainage_efficiency": drainage_eff,
        "water_depth_m": water_depth
    })

def train_and_save():
    os.makedirs("models", exist_ok=True)
    df = create_hydrological_training_data()
    
    features = [
        "elevation_meters", "rain_72h_mm", "peak_hourly_rain_mm",
        "soil_saturation", "river_discharge_m3s", "drainage_efficiency"
    ]
    X = df[features]
    y = df["water_depth_m"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(f"Model Trained! Test RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.3f} m | R2: {r2_score(y_test, preds):.3f}")
    
    joblib.dump(model, "models/flood_model.pkl")
    print("Saved model artifact to models/flood_model.pkl")

if __name__ == "__main__":
    train_and_save()