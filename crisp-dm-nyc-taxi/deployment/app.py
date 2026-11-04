from typing import Optional
from pathlib import Path
import math
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="NYC Taxi Duration Prototype", version="1.0")
MODEL_PATH = Path(__file__).with_name("model.joblib")
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class TripRequest(BaseModel):
    vendor_id: int = Field(1, ge=1, le=3)
    pickup_latitude: float = Field(..., ge=-90, le=90)
    pickup_longitude: float = Field(..., ge=-180, le=180)
    dropoff_latitude: Optional[float] = Field(None, ge=-90, le=90)
    dropoff_longitude: Optional[float] = Field(None, ge=-180, le=180)
    passenger_count: int = Field(1, ge=1, le=8)
    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_weekday: int = Field(0, ge=0, le=6)
    pickup_month: int = Field(1, ge=1, le=12)
    store_and_fwd_flag: bool = False

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}

@app.post("/predict")
def predict(trip: TripRequest):
    if bundle is None:
        return {"estimated_duration_seconds": None, "status": "model-artifact-required"}
    lat1, lat2 = math.radians(trip.pickup_latitude), math.radians(trip.dropoff_latitude or trip.pickup_latitude)
    lon1, lon2 = math.radians(trip.pickup_longitude), math.radians(trip.dropoff_longitude or trip.pickup_longitude)
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    distance = 2*6371.0088*math.asin(math.sqrt(a))
    row = {
        "vendor_id": trip.vendor_id, "passenger_count": trip.passenger_count,
        "pickup_longitude": trip.pickup_longitude, "pickup_latitude": trip.pickup_latitude,
        "dropoff_longitude": trip.dropoff_longitude or trip.pickup_longitude,
        "dropoff_latitude": trip.dropoff_latitude or trip.pickup_latitude,
        "store_and_fwd_flag": int(trip.store_and_fwd_flag), "pickup_hour": trip.pickup_hour,
        "pickup_weekday": trip.pickup_weekday, "pickup_month": trip.pickup_month,
        "hour_sin": math.sin(2*math.pi*trip.pickup_hour/24), "hour_cos": math.cos(2*math.pi*trip.pickup_hour/24),
        "weekday_sin": math.sin(2*math.pi*trip.pickup_weekday/7), "weekday_cos": math.cos(2*math.pi*trip.pickup_weekday/7),
        "haversine_km": distance,
    }
    import pandas as pd
    pred = max(0.0, math.expm1(float(bundle["model"].predict(pd.DataFrame([row])[bundle["features"]])[0])))
    return {"estimated_duration_seconds": round(pred, 2), "estimated_duration_minutes": round(pred/60, 2), "status": "ok"}
