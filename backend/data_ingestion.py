# backend/data_ingestion.py
import requests
import numpy as np
import pandas as pd
import os

CHENNAI_ZONES = [
    {"id": "CHN_01", "name": "Velachery Lowlands", "lat": 12.9791, "lon": 80.2185, "base_drainage": 0.25},
    {"id": "CHN_02", "name": "Saidapet (Adyar Basin)", "lat": 13.0213, "lon": 80.2231, "base_drainage": 0.40},
    {"id": "CHN_03", "name": "T. Nagar Basin", "lat": 13.0418, "lon": 80.2341, "base_drainage": 0.50},
    {"id": "CHN_04", "name": "Guindy High Ground", "lat": 13.0067, "lon": 80.2025, "base_drainage": 0.75},
    {"id": "CHN_05", "name": "Kotturpuram River Edge", "lat": 13.0165, "lon": 80.2435, "base_drainage": 0.30}
]

def fetch_real_elevation(lat: float, lon: float) -> float:
    """Fetch elevation in meters from Open-Meteo elevation API with fallback."""
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url, timeout=5).json()
        elev = res.get("elevation", [10.0])
        if isinstance(elev, list) and len(elev) > 0 and elev[0] is not None:
            return float(elev[0])
        return 10.0
    except Exception:
        return 10.0

def fetch_weather_forecast(lat: float, lon: float) -> dict:
    """Fetch 72h precipitation & soil moisture series."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"hourly=precipitation,soil_moisture_0_to_1cm&"
            f"forecast_days=3&timezone=Asia%2FKolkata"
        )
        res = requests.get(url, timeout=5).json()
        hourly = res.get("hourly", {})
        precip = [float(p) if p is not None else 0.0 for p in hourly.get("precipitation", [])]
        soil = [float(s) if s is not None else 0.35 for s in hourly.get("soil_moisture_0_to_1cm", [])]
        
        if not precip:
            precip = [1.5] * 72
        if not soil:
            soil = [0.35] * 72

        return {
            "hourly_precipitation": precip,
            "hourly_soil_moisture": soil,
            "rain_24h_mm": sum(precip[:24]),
            "rain_48h_mm": sum(precip[:48]),
            "rain_72h_mm": sum(precip[:72]),
            "peak_hourly_rain_mm": max(precip[:72]) if precip else 0.0,
            "mean_soil_saturation": float(np.mean(soil[:72])) if soil else 0.35
        }
    except Exception:
        return {
            "hourly_precipitation": [1.5] * 72,
            "hourly_soil_moisture": [0.35] * 72,
            "rain_24h_mm": 36.0,
            "rain_48h_mm": 72.0,
            "rain_72h_mm": 108.0,
            "peak_hourly_rain_mm": 3.0,
            "mean_soil_saturation": 0.35
        }

def inject_cyclone_scenario(weather_dict: dict) -> dict:
    """Mutates forecast dict to inject extreme storm conditions."""
    weather_dict["hourly_precipitation"] = [4.5] * 72
    weather_dict["hourly_soil_moisture"] = [0.88] * 72
    weather_dict["rain_72h_mm"] = 310.0
    weather_dict["mean_soil_saturation"] = 0.88
    return weather_dict

def fetch_river_discharge(lat: float, lon: float) -> float:
    """Safely fetch river discharge (m³/s), guarding against None/NaN."""
    try:
        url = f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}&daily=river_discharge&forecast_days=3"
        res = requests.get(url, timeout=5).json()
        daily = res.get("daily", {})
        discharge_list = daily.get("river_discharge", [2.5])
        if discharge_list and discharge_list[0] is not None:
            return float(discharge_list[0])
        return 2.5
    except Exception:
        return 2.5

def extract_lead_time_weather(weather_dict: dict, lead_hours: int = 72) -> dict:
    """Safely slices rainfall & moisture up to specified hour."""
    hours = max(1, min(int(lead_hours), 72))
    
    precip_slice = weather_dict.get("hourly_precipitation", [])[:hours]
    soil_slice = weather_dict.get("hourly_soil_moisture", [])[:hours]
    
    accumulated_rain = float(sum(precip_slice)) if precip_slice else 0.0
    peak_rain = float(max(precip_slice)) if precip_slice else 0.0
    avg_soil = float(np.mean(soil_slice)) if soil_slice else 0.35
    
    return {
        "lead_time_hours": hours,
        "accumulated_rain_mm": round(accumulated_rain, 2),
        "peak_hourly_rain_mm": round(peak_rain, 2),
        "soil_saturation": round(avg_soil, 2)
    }

def build_live_dataset():
    """Generates the live IoT / environmental snapshot CSV."""
    records = []
    for zone in CHENNAI_ZONES:
        elevation = fetch_real_elevation(zone["lat"], zone["lon"])
        weather = fetch_weather_forecast(zone["lat"], zone["lon"])
        discharge = fetch_river_discharge(zone["lat"], zone["lon"])
        
        soil_factor = 1.0 + weather["mean_soil_saturation"]
        net_water_load = weather["rain_72h_mm"] * (1.0 - zone["base_drainage"]) * soil_factor
        water_depth_m = max(0.0, (net_water_load * 0.02 + discharge * 0.12) - (elevation * 0.04))
        
        records.append({
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "elevation_meters": elevation,
            "rain_72h_mm": weather["rain_72h_mm"],
            "peak_hourly_rain_mm": weather["peak_hourly_rain_mm"],
            "soil_saturation": weather["mean_soil_saturation"],
            "river_discharge_m3s": discharge,
            "drainage_efficiency": zone["base_drainage"],
            "predicted_water_depth_m": round(water_depth_m, 2)
        })
    
    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/live_iot_stream.csv", index=False)
    print("Exported data/live_iot_stream.csv successfully.")
    return df

if __name__ == "__main__":
    build_live_dataset()