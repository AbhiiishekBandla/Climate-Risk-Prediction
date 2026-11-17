"""Data Collection and Ingestion Module for Climate Risk Prediction System.

Fetches real-world historical daily meteorological observations for Indian climate zones
from the Open-Meteo Historical Weather API (derived from ECMWF ERA5 reanalysis and IMD/WMO stations).
License: Creative Commons Attribution 4.0 International (CC BY 4.0).
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests
import pandas as pd
from typing import Dict, List, Any
from src.utils.logger import logger
from src.utils.common import read_yaml

CITIES = [
    {"city": "New Delhi", "state": "Delhi", "latitude": 28.6139, "longitude": 77.2090, "climate_zone": "Composite/Semi-Arid"},
    {"city": "Mumbai", "state": "Maharashtra", "latitude": 19.0760, "longitude": 72.8777, "climate_zone": "Coastal/Tropical Wet"},
    {"city": "Kolkata", "state": "West Bengal", "latitude": 22.5726, "longitude": 88.3639, "climate_zone": "Humid Subtropical/Gangetic"},
    {"city": "Chennai", "state": "Tamil Nadu", "latitude": 13.0827, "longitude": 80.2707, "climate_zone": "Tropical Maritime/Coromandel"}
]

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "surface_pressure_mean",
    "relative_humidity_2m_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration"
]

def fetch_city_historical_data(city_info: Dict[str, Any], start_date: str = "2015-01-01", end_date: str = "2023-12-31", max_retries: int = 3) -> pd.DataFrame:
    """Fetches real historical climate data for a given location from Open-Meteo API with retry."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city_info["latitude"],
        "longitude": city_info["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Kolkata"
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Fetching historical observations for {city_info['city']} (Attempt {attempt}/{max_retries})...")
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                wait_time = attempt * 5
                logger.warning(f"Rate limited (429). Backing off for {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            
            data = response.json()
            daily_data = data.get("daily", {})
            df = pd.DataFrame(daily_data)
            
            # Add geographical and metadata tags
            df["city"] = city_info["city"]
            df["state"] = city_info["state"]
            df["latitude"] = city_info["latitude"]
            df["longitude"] = city_info["longitude"]
            df["climate_zone"] = city_info["climate_zone"]
            
            logger.info(f"Successfully retrieved {len(df)} records for {city_info['city']}.")
            return df
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed fetching data for {city_info['city']} after {max_retries} attempts: {e}")
                raise e
            time.sleep(attempt * 3)

def collect_historical_climate_data(output_path: str = "data/raw/climate_risk_data.csv",
                                    start_date: str = "2015-01-01",
                                    end_date: str = "2023-12-31",
                                    force_download: bool = False) -> pd.DataFrame:
    """Collects multi-city historical climate dataset, validates, and stores raw data.
    
    Reuses existing raw dataset if already collected to avoid hitting API rate limits.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    provenance_path = os.path.join(os.path.dirname(output_path), "dataset_provenance.json")
    
    # If already collected and not forcing, validate and return existing
    if os.path.exists(output_path) and not force_download:
        existing_df = pd.read_csv(output_path)
        if len(existing_df) >= 10000:
            logger.info(f"Using verified existing dataset from {output_path} ({len(existing_df)} records).")
            # Ensure provenance file is present
            if not os.path.exists(provenance_path):
                provenance = {
                    "dataset_name": "Indian Historical Multi-Zone Climate Dataset",
                    "source": "Open-Meteo Historical Weather API (ERA5 Reanalysis & IMD/WMO ground stations)",
                    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_records": len(existing_df),
                    "cities": [c["city"] for c in CITIES]
                }
                with open(provenance_path, "w", encoding="utf-8") as f:
                    json.dump(provenance, f, indent=2)
            return existing_df
            
    all_dfs = []
    for city in CITIES:
        df_city = fetch_city_historical_data(city, start_date=start_date, end_date=end_date)
        all_dfs.append(df_city)
        time.sleep(1.0)
            
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df["time"] = pd.to_datetime(combined_df["time"])
    combined_df = combined_df.sort_values(by=["time", "city"]).reset_index(drop=True)
    
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Dataset successfully compiled: {len(combined_df)} rows, {len(combined_df.columns)} columns saved to {output_path}")
    
    provenance = {
        "dataset_name": "Indian Historical Multi-Zone Climate Dataset",
        "source": "Open-Meteo Historical Weather API (ERA5 Reanalysis & IMD/WMO ground stations)",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "citation": "Open-Meteo.com, Copernicus Climate Change Service (ERA5)",
        "start_date": start_date,
        "end_date": end_date,
        "total_records": len(combined_df),
        "cities": [c["city"] for c in CITIES],
        "variables": DAILY_VARIABLES
    }
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"Saved dataset provenance to {provenance_path}")
    
    return combined_df

if __name__ == "__main__":
    force = "--force" in sys.argv
    collect_historical_climate_data(force_download=force)
