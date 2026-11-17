"""Feature Engineering and Preprocessing Pipeline for Climate Risk Prediction.

Calculates meteorologically sound indicators:
- Heat Index (apparent thermal stress)
- Diurnal Temperature Range (DTR)
- Multi-day cumulative precipitation (3-day and 7-day rolling sums)
- Consecutive dry days (drought risk)
- Pressure gradient & wind gust factor
- Season extraction according to IMD meteorological calendar
- Derived proxy climate risk target labels (Low: 0, Moderate: 1, High: 2)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from src.utils.logger import logger

def calculate_heat_index(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Calculates apparent temperature / Heat Index using NOAA/Rothfusz regression formula."""
    T = temp_c * 9/5 + 32  # Convert to Fahrenheit
    RH = rh_pct
    
    # Simple formula
    hi = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    
    # Rothfusz full regression for warmer conditions (T >= 80 F)
    full_hi = (-42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH -
               0.00683783*T**2 - 0.05481717*RH**2 + 0.00122874*T**2*RH +
               0.00085282*T*RH**2 - 0.00000199*T**2*RH**2)
    
    heat_index_f = np.where(T >= 80.0, full_hi, hi)
    heat_index_c = (heat_index_f - 32) * 5/9
    return pd.Series(heat_index_c, index=temp_c.index).clip(lower=-20.0, upper=65.0)

def assign_imd_season(month: int) -> str:
    """Categorizes month into standard India Meteorological Department (IMD) seasons."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer/Pre-Monsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

def derive_risk_labels(df: pd.DataFrame) -> pd.Series:
    """Derives meteorological proxy risk level based on published IMD & NDMA operational criteria:
    
    - Level 0 (Low / Normal): Benign conditions.
    - Level 1 (Moderate Risk):
        * Moderate heat stress (Heat Index >= 38.0 C or T_max >= 39.0 C)
        * Moderate rainfall (Precipitation >= 30.0 mm/day or 3-day sum >= 60.0 mm)
        * Elevated winds (Wind gusts >= 40.0 km/h)
    - Level 2 (High / Severe Risk):
        * Extreme Heatwave (T_max >= 42.0 C or Heat Index >= 45.0 C)
        * Heavy Rainfall / Flood Risk (Precipitation >= 64.5 mm [IMD Heavy Rain criterion] or 3-day sum >= 120.0 mm)
        * Severe Storm / Gale Winds (Wind gusts >= 60.0 km/h)
    
    NOTE: These are clearly documented meteorological proxy labels based on severity thresholds.
    """
    is_high_heat = (df["temperature_2m_max"] >= 42.0) | (df["heat_index"] >= 45.0)
    is_high_rain = (df["precipitation_sum"] >= 64.5) | (df["precip_3d_sum"] >= 120.0)
    is_high_wind = (df["windgusts_10m_max"] >= 60.0)
    high_risk = is_high_heat | is_high_rain | is_high_wind
    
    is_mod_heat = (df["temperature_2m_max"] >= 39.0) | (df["heat_index"] >= 38.0)
    is_mod_rain = (df["precipitation_sum"] >= 30.0) | (df["precip_3d_sum"] >= 60.0)
    is_mod_wind = (df["windgusts_10m_max"] >= 40.0)
    moderate_risk = (is_mod_heat | is_mod_rain | is_mod_wind) & (~high_risk)
    
    risk = np.zeros(len(df), dtype=int)
    risk[moderate_risk] = 1
    risk[high_risk] = 2
    return pd.Series(risk, index=df.index, name="risk_level")

def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Performs full feature engineering on the climate dataset."""
    df = raw_df.copy()
    
    # Ensure datetime
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by=["city", "time"]).reset_index(drop=True)
    
    # 1. Temporal features
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["day_of_year"] = df["time"].dt.dayofyear
    df["season"] = df["month"].apply(assign_imd_season)
    
    # 2. Thermal indicators
    df["diurnal_temp_range"] = (df["temperature_2m_max"] - df["temperature_2m_min"]).clip(lower=0.0)
    df["heat_index"] = calculate_heat_index(df["temperature_2m_max"], df["relative_humidity_2m_mean"])
    df["apparent_temp_diff"] = df["apparent_temperature_max"] - df["temperature_2m_max"]
    
    # 3. Moisture & Hydrological indicators (computed grouped by city)
    df["precip_3d_sum"] = df.groupby("city")["precipitation_sum"].transform(
        lambda s: s.rolling(window=3, min_periods=1).sum()
    )
    df["precip_7d_sum"] = df.groupby("city")["precipitation_sum"].transform(
        lambda s: s.rolling(window=7, min_periods=1).sum()
    )
    
    # Consecutive dry days (< 1mm precipitation)
    def calc_dry_days(s: pd.Series) -> pd.Series:
        is_dry = (s < 1.0).astype(int)
        # Cumulative sum reset when rain occurs
        streak = is_dry * (is_dry.groupby((~is_dry.astype(bool)).cumsum()).cumcount() + 1)
        return streak

    df["dry_spell_days"] = df.groupby("city")["precipitation_sum"].transform(calc_dry_days)
    
    # 4. Wind & Dynamics
    df["wind_gust_factor"] = (df["windgusts_10m_max"] / (df["windspeed_10m_max"] + 1e-4)).clip(lower=1.0, upper=5.0)
    
    # 5. Evaporative stress
    df["water_deficit"] = (df["et0_fao_evapotranspiration"] - df["precipitation_sum"]).clip(lower=-150.0, upper=30.0)
    
    # 6. Proxy Climate Risk Target
    df["risk_level"] = derive_risk_labels(df)
    
    logger.info(f"Feature engineering completed: {df.shape[1]} columns. Risk distribution:")
    logger.info(f"{df['risk_level'].value_counts(normalize=True).to_dict()}")
    
    return df

def clean_and_split_data(raw_path: str = "data/raw/climate_risk_data.csv",
                         train_path: str = "data/processed/train.csv",
                         test_path: str = "data/processed/test.csv",
                         split_year: int = 2022) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cleans data, engineers features, and performs temporal train/test split.
    
    Temporal Split Strategy:
    - Train Set: 2015-01-01 to 2021-12-31 (Past historical period)
    - Test Set: 2022-01-01 to 2023-12-31 (Future out-of-time evaluation period)
    This strictly prevents temporal data leakage.
    """
    logger.info(f"Loading raw data from {raw_path}...")
    raw_df = pd.read_csv(raw_path)
    
    # Handle duplicates
    init_len = len(raw_df)
    raw_df = raw_df.drop_duplicates(subset=["time", "city"]).reset_index(drop=True)
    if len(raw_df) < init_len:
        logger.info(f"Removed {init_len - len(raw_df)} duplicate rows.")
        
    # Handle missing values using forward/backward fill per city
    raw_df["time"] = pd.to_datetime(raw_df["time"])
    raw_df = raw_df.sort_values(by=["city", "time"]).reset_index(drop=True)
    
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
    raw_df[numeric_cols] = raw_df.groupby("city")[numeric_cols].transform(lambda grp: grp.ffill().bfill())
    
    # Apply feature engineering
    processed_df = engineer_features(raw_df)
    
    # Temporal train/test split
    train_df = processed_df[processed_df["year"] < split_year].copy().reset_index(drop=True)
    test_df = processed_df[processed_df["year"] >= split_year].copy().reset_index(drop=True)
    
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Temporal Split: Train={len(train_df)} rows ({train_df['year'].min()}-{train_df['year'].max()}), "
                f"Test={len(test_df)} rows ({test_df['year'].min()}-{test_df['year'].max()})")
    logger.info(f"Train saved to {train_path}, Test saved to {test_path}")
    
    # Also save reference and current datasets for future drift monitoring (Phase 4)
    ref_path = "data/reference/reference_data.csv"
    curr_path = "data/current/current_data.csv"
    Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
    Path(curr_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(ref_path, index=False)
    test_df.to_csv(curr_path, index=False)
    logger.info(f"Reference and Current datasets prepared for Evidently drift monitoring.")
    
    return train_df, test_df

if __name__ == "__main__":
    clean_and_split_data()
