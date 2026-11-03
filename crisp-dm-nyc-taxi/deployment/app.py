from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="NYC Taxi Duration Prototype", version="1.0")

class TripRequest(BaseModel):
    pickup_latitude: float = Field(..., ge=-90, le=90)
    pickup_longitude: float = Field(..., ge=-180, le=180)
    dropoff_latitude: Optional[float] = Field(None, ge=-90, le=90)
    dropoff_longitude: Optional[float] = Field(None, ge=-180, le=180)
    passenger_count: int = Field(1, ge=1, le=8)
    pickup_hour: int = Field(..., ge=0, le=23)

@app.get("/health")
def health():
    return {"status": "ok", "model": "documented-prototype"}

@app.post("/predict")
def predict(trip: TripRequest):
    # Replace with the serialized, validated model after notebook execution.
    return {"estimated_duration_seconds": None, "status": "model-artifact-required"}
