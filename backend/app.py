# app.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from datetime import datetime
from data_ingestion import CHENNAI_ZONES, fetch_real_elevation, fetch_weather_forecast, fetch_river_discharge

app = FastAPI(title="Chennai Flood Prediction Engine", version="1.0")

# Enable CORS for Frontend (Person 3)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
model = joblib.load("models/flood_model.pkl")

SAFE_SHELTERS = [
    {
        "shelter_id": "SHELTER_CHN_01",
        "name": "St. Thomas Mount Relief Center",
        "coordinates": [80.193, 13.005],
        "elevation_meters": 62.0,
        "capacity": {"max": 1200, "current_occupancy": 310, "available_space": 890},
        "accessibility_score": 0.95
    },
    {
        "shelter_id": "SHELTER_CHN_02",
        "name": "Anna University Indoor Complex",
        "coordinates": [80.235, 13.013],
        "elevation_meters": 22.0,
        "capacity": {"max": 800, "current_occupancy": 150, "available_space": 650},
        "accessibility_score": 0.88
    }
]

# Polygon bounding boxes for map overlays
BOUNDARIES = {
    "CHN_01": [[[80.210, 12.970], [80.230, 12.970], [80.230, 12.990], [80.210, 12.990], [80.210, 12.970]]],
    "CHN_02": [[[80.215, 13.010], [80.235, 13.010], [80.235, 13.030], [80.215, 13.030], [80.215, 13.010]]],
    "CHN_03": [[[80.230, 13.030], [80.250, 13.030], [80.250, 13.050], [80.230, 13.050], [80.230, 13.030]]],
    "CHN_04": [[[80.195, 13.000], [80.215, 13.000], [80.215, 13.020], [80.195, 13.020], [80.195, 13.000]]],
    "CHN_05": [[[80.235, 13.015], [80.255, 13.015], [80.255, 13.035], [80.235, 13.035], [80.235, 13.015]]]
}

def classify_risk(depth_m):
    if depth_m >= 0.60:
        return {"risk_level": "CRITICAL", "is_passable_for_vehicles": False, "urgency": "IMMEDIATE"}
    elif depth_m >= 0.20:
        return {"risk_level": "MODERATE", "is_passable_for_vehicles": True, "urgency": "WARNING"}
    else:
        return {"risk_level": "LOW", "is_passable_for_vehicles": True, "urgency": "SAFE"}

@app.get("/api/flood-risk")
def get_flood_risk(scenario: str = Query("live", enum=["live", "heavy_cyclone", "extreme_breach"])):
    zones_output = []
    
    for zone in CHENNAI_ZONES:
        elevation = fetch_real_elevation(zone["lat"], zone["lon"])
        weather = fetch_weather_forecast(zone["lat"], zone["lon"])
        discharge = fetch_river_discharge(zone["lat"], zone["lon"])
        
        # Inject stress test multipliers for demo scenarios
        if scenario == "heavy_cyclone":
            weather["rain_72h_mm"] = 280.0
            weather["peak_hourly_rain_mm"] = 45.0
            weather["mean_soil_saturation"] = 0.85
            discharge = 25.0
        elif scenario == "extreme_breach":
            weather["rain_72h_mm"] = 420.0
            weather["peak_hourly_rain_mm"] = 70.0
            weather["mean_soil_saturation"] = 0.95
            discharge = 48.0
            
        features_df = pd.DataFrame([{
            "elevation_meters": elevation,
            "rain_72h_mm": weather["rain_72h_mm"],
            "peak_hourly_rain_mm": weather["peak_hourly_rain_mm"],
            "soil_saturation": weather["mean_soil_saturation"],
            "river_discharge_m3s": discharge,
            "drainage_efficiency": zone["base_drainage"]
        }])
        
        predicted_depth = float(model.predict(features_df)[0])
        risk_info = classify_risk(predicted_depth)
        
        zones_output.append({
            "zone_id": zone["zone_id"],
            "zone_name": zone["name"],
            "boundary": {
                "type": "Polygon",
                "coordinates": BOUNDARIES.get(zone["zone_id"], [])
            },
            "environmental_metrics": {
                "rainfall_forecast_mm": round(weather["rain_72h_mm"], 2),
                "elevation_meters": round(elevation, 2),
                "drainage_efficiency": zone["base_drainage"],
                "river_gauge_level_m": round(discharge * 0.15, 2)
            },
            "risk_assessment": {
                "predicted_water_depth_m": round(predicted_depth, 2),
                "risk_level": risk_info["risk_level"],
                "is_passable_for_vehicles": risk_info["is_passable_for_vehicles"],
                "evacuation_urgency": risk_info["urgency"]
            }
        })
        
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "lead_time_hours": 72,
        "scenario": scenario,
        "zones": zones_output,
        "safe_destinations": SAFE_SHELTERS
    }
@app.get("/")
def root():
    return {"message": "Chennai Flood Prediction Engine is running. Visit /docs or /api/flood-risk"}