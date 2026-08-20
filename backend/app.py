# backend/app.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import traceback
import os

from data_ingestion import (
    CHENNAI_ZONES,
    fetch_real_elevation,
    fetch_weather_forecast,
    fetch_river_discharge,
    extract_lead_time_weather
)

# Safe fallback for alerts import
try:
    from alerts import generate_localized_alert
except ImportError:
    ALERT_TEMPLATES = {
        "Red": {
            "en": "EMERGENCY: Critical flood risk in {zone} ({depth}m expected). Evacuate to: {shelter}.",
            "ta": "அவசர எச்சரிக்கை: {zone} பகுதியில் அதிக வெள்ள அபாயம் ({depth}m நீர் மட்டம்). பாதுகாப்பான இடம்: {shelter}."
        },
        "Orange": {
            "en": "WARNING: Moderate waterlogging in {zone} ({depth}m expected). Avoid transit.",
            "ta": "எச்சரிக்கை: {zone} பகுதியில் மிதமான நீர் தேக்கம் ({depth}m). தாழ்வான சாலைகளை தவிர்க்கவும்."
        },
        "Green": {
            "en": "ADVISORY: Normal conditions in {zone}.",
            "ta": "அறிவிப்பு: {zone} பகுதியில் நிலைமை சீராக உள்ளது."
        }
    }
    def generate_localized_alert(zone_name: str, risk_level: str, depth_m: float, lead_time: int = 72, shelter: str = "Anna University (Guindy)"):
        templates = ALERT_TEMPLATES.get(risk_level, ALERT_TEMPLATES["Green"])
        return {
            "en": templates["en"].format(zone=zone_name, depth=round(depth_m, 2), lead_time=lead_time, shelter=shelter),
            "ta": templates["ta"].format(zone=zone_name, depth=round(depth_m, 2), lead_time=lead_time, shelter=shelter)
        }

app = FastAPI(title="Chennai Flood Early Warning API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model safely across varying execution paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_candidates = [
    os.path.join(base_dir, "models", "flood_model.pkl"),
    os.path.join(base_dir, "flood_model.pkl"),
    "models/flood_model.pkl",
    "flood_model.pkl"
]

model = None
for path in model_candidates:
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            print(f"Loaded ML model successfully from: {path}")
            break
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")

SAFE_SHELTERS = [
    {
        "shelter_id": "SHELTER_GUINDY_CAMPUS",
        "name": "Anna University Shelter (Guindy)",
        "latitude": 13.0100,
        "longitude": 80.2350,
        "elevation_m": 22.0,
        "capacity_people": 3000,
        "current_occupancy": 450,
        "is_operational": True
    },
    {
        "shelter_id": "SHELTER_ST_THOMAS",
        "name": "St. Thomas Relief Center",
        "latitude": 13.0040,
        "longitude": 80.1930,
        "elevation_m": 60.0,
        "capacity_people": 1800,
        "current_occupancy": 200,
        "is_operational": True
    }
]

BOUNDARIES = {
    "CHN_01": [[80.2085, 12.9691], [80.2285, 12.9691], [80.2285, 12.9891], [80.2085, 12.9891], [80.2085, 12.9691]],
    "CHN_02": [[80.2131, 13.0113], [80.2331, 13.0113], [80.2331, 13.0313], [80.2131, 13.0313], [80.2131, 13.0113]],
    "CHN_03": [[80.2241, 13.0318], [80.2441, 13.0318], [80.2441, 13.0518], [80.2241, 13.0518], [80.2241, 13.0318]],
    "CHN_04": [[80.1925, 12.9967], [80.2125, 12.9967], [80.2125, 13.0167], [80.1925, 13.0167], [80.1925, 12.9967]],
    "CHN_05": [[80.2335, 13.0065], [80.2535, 13.0065], [80.2535, 13.0265], [80.2335, 13.0265], [80.2335, 13.0065]]
}

def classify_risk(depth_m: float):
    if depth_m >= 0.60:
        return {"risk_level": "Red", "is_passable_for_vehicles": False, "urgency": "IMMEDIATE"}
    elif depth_m >= 0.20:
        return {"risk_level": "Orange", "is_passable_for_vehicles": True, "urgency": "WARNING"}
    else:
        return {"risk_level": "Green", "is_passable_for_vehicles": True, "urgency": "SAFE"}

def predict_depth_fallback(features_dict: dict) -> float:
    """Hydrological physics fallback if ML model artifact fails to evaluate."""
    net_load = features_dict["rain_72h_mm"] * (1.0 - features_dict["drainage_efficiency"]) * (1.0 + features_dict["soil_saturation"])
    depth = (net_load * 0.015 + features_dict["river_discharge_m3s"] * 0.08) - (features_dict["elevation_meters"] * 0.05)
    return max(0.0, float(depth))

@app.get("/api/flood-risk")
def get_flood_risk(
    scenario: str = Query("live", description="live, heavy_cyclone, extreme_breach"),
    lead_time_hours: int = Query(72, ge=1, le=72, description="Forecast window +1h to +72h")
):
    try:
        results = []

        for zone in CHENNAI_ZONES:
            # 1. Fetch baselines safely
            elevation = float(fetch_real_elevation(zone["lat"], zone["lon"]))
            weather_raw = fetch_weather_forecast(zone["lat"], zone["lon"])
            discharge = float(fetch_river_discharge(zone["lat"], zone["lon"]))

            weather_lead = extract_lead_time_weather(weather_raw, lead_hours=lead_time_hours)
            rain_lead_mm = float(weather_lead.get("accumulated_rain_mm", 0.0))
            peak_rain_mm = float(weather_lead.get("peak_hourly_rain_mm", 0.0))
            soil_sat = float(weather_lead.get("soil_saturation", 0.4))

            # 2. Scenarios
            if scenario == "heavy_cyclone":
                rain_lead_mm = round((280.0 / 72.0) * lead_time_hours, 2)
                peak_rain_mm = 35.0
                soil_sat = 0.88
                discharge = 25.0
            elif scenario == "extreme_breach":
                rain_lead_mm = round((420.0 / 72.0) * lead_time_hours, 2)
                peak_rain_mm = 55.0
                soil_sat = 0.95
                discharge = 45.0

            feature_dict = {
                "elevation_meters": elevation,
                "rain_72h_mm": rain_lead_mm,
                "peak_hourly_rain_mm": peak_rain_mm,
                "soil_saturation": soil_sat,
                "river_discharge_m3s": discharge,
                "drainage_efficiency": float(zone.get("base_drainage", 0.4))
            }

            # 3. Model inference with DataFrame wrapper to match Scikit-Learn training schema
            if model is not None:
                features_df = pd.DataFrame([feature_dict])
                predicted_depth = float(model.predict(features_df)[0])
            else:
                predicted_depth = predict_depth_fallback(feature_dict)

            predicted_depth = max(0.0, round(predicted_depth, 2))
            risk_info = classify_risk(predicted_depth)

            results.append({
                "zone_id": zone["id"],
                "name": zone["name"],
                "latitude": zone["lat"],
                "longitude": zone["lon"],
                "elevation_m": elevation,
                "lead_time_hours": lead_time_hours,
                "accumulated_rain_mm": rain_lead_mm,
                "peak_hourly_rain_mm": peak_rain_mm,
                "soil_saturation": soil_sat,
                "river_discharge_m3s": discharge,
                "drainage_efficiency": zone.get("base_drainage", 0.4),
                "predicted_depth_m": predicted_depth,
                "risk": risk_info,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [BOUNDARIES.get(zone["id"], [])]
                }
            })

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "lead_time_hours": lead_time_hours,
            "scenario": scenario,
            "zones": results,
            "safe_destinations": SAFE_SHELTERS
        }

    except Exception as e:
        print("ERROR IN /api/flood-risk:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction Engine Error: {str(e)}")

@app.get("/api/routing/blocked-nodes")
def get_blocked_nodes(
    scenario: str = Query("live"),
    lead_time_hours: int = Query(72, ge=1, le=72)
):
    risk_data = get_flood_risk(scenario=scenario, lead_time_hours=lead_time_hours)
    blocked_zones = [
        {
            "zone_id": z["zone_id"],
            "zone_name": z["name"],
            "coordinates": [z["latitude"], z["longitude"]],
            "predicted_depth_m": z["predicted_depth_m"],
            "risk_level": z["risk"]["risk_level"]
        }
        for z in risk_data["zones"]
        if not z["risk"]["is_passable_for_vehicles"]
    ]
    return {
        "scenario": scenario,
        "lead_time_hours": lead_time_hours,
        "blocked_count": len(blocked_zones),
        "blocked_zones": blocked_zones
    }

@app.get("/api/alerts")
def get_emergency_alerts(
    scenario: str = Query("live", description="Scenario type: live, heavy_cyclone, extreme_breach"),
    lead_time_hours: int = Query(72, ge=1, le=72)
):
    try:
        risk_data = get_flood_risk(scenario=scenario, lead_time_hours=lead_time_hours)
        alerts = []
        
        for z in risk_data.get("zones", []):
            risk_level = z.get("risk", {}).get("risk_level", "LOW")
            
            # Trigger alerts for Red and Orange tiers
            if risk_level in ["Red", "Orange", "CRITICAL", "MODERATE"]:
                # Normalize risk name if needed
                normalized_level = "Red" if risk_level in ["Red", "CRITICAL"] else "Orange"
                depth = float(z.get("predicted_depth_m", 0.0))
                zone_name = z.get("name", "Unknown Zone")

                messages = generate_localized_alert(
                    zone_name=zone_name,
                    risk_level=normalized_level,
                    depth_m=depth,
                    lead_time=lead_time_hours
                )

                alerts.append({
                    "zone_id": z.get("zone_id"),
                    "zone_name": zone_name,
                    "risk_level": normalized_level,
                    "predicted_depth_m": depth,
                    "messages": messages
                })
                
        return {
            "scenario": scenario,
            "lead_time_hours": lead_time_hours,
            "active_alerts_count": len(alerts),
            "alerts": alerts
        }
    except Exception as e:
        print("ERROR IN /api/alerts:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Alert Dispatch Error: {str(e)}")
@app.get("/api/shelters")
def get_shelters():
    return {"total_shelters": len(SAFE_SHELTERS), "shelters": SAFE_SHELTERS}

@app.get("/")
def root():
    return {"message": "Chennai Flood Prediction Engine is Active. Visit /docs"}