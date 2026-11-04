"""CRISP-DM data-preparation evidence and leakage-safe feature construction."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "NYC_sample_200k.csv"
OUT = ROOT / "artifacts" / "data_preparation"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(DATA)
df = raw.copy()
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

audit = pd.DataFrame({"id": df["id"], "source_row": np.arange(len(df))})
audit["reason"] = "retained"
audit["flag_zero_passengers"] = df["passenger_count"].eq(0)
audit["flag_long_duration"] = df["trip_duration"].ge(7200)
audit["flag_boundary_duration"] = df["trip_duration"].ge(80000)
audit["flag_outside_display_window"] = (
    (df["pickup_longitude"] < -74.30) | (df["pickup_longitude"] > -73.65) |
    (df["pickup_latitude"] < 40.45) | (df["pickup_latitude"] > 40.95)
)

rules = [
    ("invalid_timestamp", df["pickup_datetime"].isna() | df["dropoff_datetime"].isna()),
    ("invalid_duration", ~df["trip_duration"].between(60, 7200)),
    ("invalid_passenger_count", ~df["passenger_count"].between(1, 8)),
    ("nonfinite_coordinates", ~np.isfinite(df[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]]).all(axis=1)),
]
for reason, mask in rules:
    audit.loc[(audit["reason"] == "retained") & mask, "reason"] = reason

audit.to_csv(OUT / "exclusion_audit.csv", index=False)
clean = df[audit["reason"].eq("retained")].copy()

def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    x1, y1, x2, y2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dx, dy = x2 - x1, y2 - y1
    a = np.sin(dy / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

clean["pickup_hour"] = clean["pickup_datetime"].dt.hour
clean["pickup_weekday"] = clean["pickup_datetime"].dt.weekday
clean["pickup_month"] = clean["pickup_datetime"].dt.month
clean["hour_sin"] = np.sin(2 * np.pi * clean["pickup_hour"] / 24)
clean["hour_cos"] = np.cos(2 * np.pi * clean["pickup_hour"] / 24)
clean["weekday_sin"] = np.sin(2 * np.pi * clean["pickup_weekday"] / 7)
clean["weekday_cos"] = np.cos(2 * np.pi * clean["pickup_weekday"] / 7)
clean["haversine_km"] = haversine_km(clean["pickup_longitude"], clean["pickup_latitude"], clean["dropoff_longitude"], clean["dropoff_latitude"])
clean["log_trip_duration"] = np.log1p(clean["trip_duration"])

predictor_columns = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "store_and_fwd_flag", "pickup_hour", "pickup_weekday", "pickup_month", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "haversine_km"]
target_columns = ["trip_duration", "log_trip_duration", "dropoff_datetime"]
leakage_overlap = sorted(set(predictor_columns) & set(target_columns))

pd.DataFrame({"metric": ["raw_rows", "retained_rows", "excluded_rows", "retained_percent", "predictor_target_overlap"], "value": [len(df), len(clean), len(df)-len(clean), round(100*len(clean)/len(df), 4), ",".join(leakage_overlap) or "none"]}).to_csv(OUT / "preparation_summary.csv", index=False)
pd.DataFrame({"predictor": predictor_columns}).to_csv(OUT / "predictor_allowlist.csv", index=False)
clean[predictor_columns + ["trip_duration", "pickup_datetime", "id"]].to_csv(OUT / "prepared_features.csv", index=False)

print(pd.Series(audit["reason"].value_counts(), name="rows").to_string())
print(f"retained={len(clean)} excluded={len(df)-len(clean)} overlap={leakage_overlap or 'none'}")
