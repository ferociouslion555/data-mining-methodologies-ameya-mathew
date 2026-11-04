"""CRISP-DM data-understanding evidence for the NYC taxi sample."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "NYC_sample_200k.csv"
OUT = ROOT / "artifacts" / "data_understanding"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["pickup_weekday"] = df["pickup_datetime"].dt.day_name()
df["pickup_month"] = df["pickup_datetime"].dt.month

quality = pd.DataFrame({
    "rows": [len(df)],
    "columns": [len(df.columns)],
    "missing_cells": [int(df.isna().sum().sum())],
    "duplicate_rows": [int(df.duplicated().sum())],
    "duplicate_ids": [int(df["id"].duplicated().sum())],
    "min_duration_seconds": [int(df["trip_duration"].min())],
    "median_duration_seconds": [float(df["trip_duration"].median())],
    "mean_duration_seconds": [float(df["trip_duration"].mean())],
    "max_duration_seconds": [int(df["trip_duration"].max())],
    "min_passenger_count": [int(df["passenger_count"].min())],
    "max_passenger_count": [int(df["passenger_count"].max())],
    "pickup_time_invalid": [int(df["pickup_datetime"].isna().sum())],
    "dropoff_time_invalid": [int(df["dropoff_datetime"].isna().sum())],
})
quality.to_csv(OUT / "quality_summary.csv", index=False)

df.describe(include="all").transpose().to_csv(OUT / "descriptive_statistics.csv")
df["pickup_weekday"].value_counts().rename_axis("weekday").reset_index(name="trips").to_csv(OUT / "weekday_counts.csv", index=False)
df.groupby("pickup_hour").agg(trips=("id", "size"), median_duration=("trip_duration", "median"), mean_duration=("trip_duration", "mean")).reset_index().to_csv(OUT / "hourly_summary.csv", index=False)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["trip_duration"], bins=100, ax=axes[0], color="#276FBF")
axes[0].set_title("Trip duration: raw scale")
axes[0].set_xlabel("Seconds")
sns.histplot(np.log1p(df["trip_duration"]), bins=100, ax=axes[1], color="#E07A5F")
axes[1].set_title("Trip duration: log1p scale")
axes[1].set_xlabel("log1p(seconds)")
fig.tight_layout()
fig.savefig(OUT / "fig_01_duration_distribution.png", dpi=160)
plt.close(fig)

hourly = df.groupby("pickup_hour", as_index=False).agg(trips=("id", "size"), median_duration=("trip_duration", "median"))
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(hourly["pickup_hour"], hourly["trips"], marker="o", color="#276FBF")
ax1.set_xlabel("Pickup hour")
ax1.set_ylabel("Trips", color="#276FBF")
ax2 = ax1.twinx()
ax2.plot(hourly["pickup_hour"], hourly["median_duration"], marker="s", color="#E07A5F")
ax2.set_ylabel("Median duration (seconds)", color="#E07A5F")
ax1.set_title("Trip volume and median duration by pickup hour")
fig.tight_layout()
fig.savefig(OUT / "fig_02_hourly_patterns.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.scatterplot(data=df.sample(min(10000, len(df)), random_state=42), x="pickup_longitude", y="pickup_latitude", s=8, alpha=.25, ax=axes[0], color="#276FBF")
axes[0].set_title("Pickup coordinate sample")
sns.scatterplot(data=df.sample(min(10000, len(df)), random_state=42), x="dropoff_longitude", y="dropoff_latitude", s=8, alpha=.25, ax=axes[1], color="#E07A5F")
axes[1].set_title("Drop-off coordinate sample")
for ax in axes:
    ax.set_xlim(-74.3, -73.65)
    ax.set_ylim(40.45, 40.95)
fig.tight_layout()
fig.savefig(OUT / "fig_03_spatial_density.png", dpi=160)
plt.close(fig)

print(quality.to_string(index=False))
print(f"Wrote evidence to {OUT}")
