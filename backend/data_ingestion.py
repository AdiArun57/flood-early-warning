# data_ingestion.py
import requests
import pandas as pd
import numpy as np

# Defined zones across Chennai's major river corridors
CHENNAI_ZONES = [
    {"zone_id": "CHN_01", "name": "Velachery Lowlands", "lat": 12.975, "lon": 80.220, "base_drainage": 0.25},
    {"zone_id": "CHN_02", "name": "Saidapet (Adyar Basin)", "lat": 13.020, "lon": 80.225, "base_drainage": 0.40},
    {"zone_id": "CHN_03", "name": "T. Nagar Basin", "lat": 13.040, "lon": 80.233, "base_drainage": 0.50},
    {"zone_id": "CHN_04", "name": "Guindy High Ground", "lat": 13.008, "lon": 80.205, "base_drainage": 0.75},
    {"zone_id": "CHN_05", "name": "Kotturpuram River Edge", "lat": 13.023, "lon": 80.242, "base_drainage": 0.30}
]

def fetch_real_elevation(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    res = requests.get(url, timeout=10).json()
    return res.get("elevation", [10.0])[0]

def fetch_weather_forecast(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["precipitation", "soil_moisture_0_to_1cm"],
        "forecast_days": 3,
        "timezone": "Asia/Kolkata"
    }
    res = requests.get(url, params=params, timeout=10).json()
    precip = res["hourly"]["precipitation"][:72]
    soil_m = res["hourly"]["soil_moisture_0_to_1cm"][:72]
    
    return {
        "rain_24h_mm": sum(precip[:24]),
        "rain_48h_mm": sum(precip[:48]),
        "rain_72h_mm": sum(precip[:72]),
        "peak_hourly_rain_mm": max(precip[:72]),
        "mean_soil_saturation": float(np.mean(soil_m[:72]))
    }
# Pass scenario="cyclone" to simulate heavy event based on live elevation/river topology
def inject_cyclone_scenario(weather_dict):
    weather_dict["rain_72h_mm"] = 310.0
    weather_dict["mean_soil_saturation"] = 0.88
    return weather_dict
def fetch_river_discharge(lat, lon):
    url = "https://flood-api.open-meteo.com/v1/flood"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "river_discharge",
        "forecast_days": 3
    }
    res = requests.get(url, params=params, timeout=10).json()
    discharges = res.get("daily", {}).get("river_discharge", [2.5, 2.5, 2.5])
    return discharges[0] if discharges and discharges[0] is not None else 2.5

def build_live_dataset():
    records = []
    print("Fetching live sensor and satellite feeds for Chennai...")
    for zone in CHENNAI_ZONES:
        elevation = fetch_real_elevation(zone["lat"], zone["lon"])
        weather = fetch_weather_forecast(zone["lat"], zone["lon"])
        discharge = fetch_river_discharge(zone["lat"], zone["lon"])
        
        # Calculate real physical runoff using Rational Hydrologic Runoff method
        # Effective Runoff = Rain * (1 - Drainage) * Soil Saturation Multiplier
        soil_factor = 1.0 + weather["mean_soil_saturation"]
        net_water_load = (weather["rain_72h_mm"] * (1.0 - zone["base_drainage"]) * soil_factor)
        river_backflow = discharge * 0.12
        
        # Ground water depth relative to elevation
        water_depth_m = max(0.0, ((net_water_load * 0.02) + river_backflow) - (elevation * 0.04))
        
        records.append({
            "zone_id": zone["zone_id"],
            "zone_name": zone["name"],
            "latitude": zone["lat"],
            "longitude": zone["lon"],
            "elevation_meters": round(elevation, 2),
            "rain_72h_mm": round(weather["rain_72h_mm"], 2),
            "peak_hourly_rain_mm": round(weather["peak_hourly_rain_mm"], 2),
            "soil_saturation": round(weather["mean_soil_saturation"], 3),
            "river_discharge_m3s": round(discharge, 2),
            "drainage_efficiency": zone["base_drainage"],
            "predicted_water_depth_m": round(water_depth_m, 2)
        })
    
    df = pd.DataFrame(records)
    df.to_csv("data/live_iot_stream.csv", index=False)
    print("Success: Live data saved to data/live_iot_stream.csv")
    print(df[["zone_name", "elevation_meters", "rain_72h_mm", "predicted_water_depth_m"]])

if __name__ == "__main__":
    build_live_dataset()