# CRISP-DM Report: NYC Taxi Trip Duration

## Business understanding
The practical objective is to estimate trip duration for passenger expectations and dispatch planning. Prediction-time features must be available before the trip ends. `dropoff_datetime` and any realized duration are therefore excluded from predictors.

## Data understanding
The target is strongly right-skewed. Hour, weekday, month, coordinates, and trip distance are plausible predictors. Extreme durations and coordinates require explicit audit rather than silent removal. Spatial plots are descriptive and do not prove causation.

## Data preparation
The pipeline parses timestamps, validates duration and coordinate domains, calculates haversine distance, encodes cyclic time variables, and fits transformations only on the training partition. Cleaning decisions are recorded in an exclusion audit.

## Modeling
Linear regression is used as a transparent baseline. Histogram gradient boosting models `log1p(trip_duration)`. Evaluation uses a January–May training period and June future holdout in addition to a random comparison.

## Evaluation
The existing experiment reported approximately 0.3334 RMSLE and 197 seconds MAE for boosting on the June holdout, versus approximately 0.4852 RMSLE for linear regression. The random split was more optimistic, which supports chronological evaluation for this use case.

## Deployment
The deployment directory contains a FastAPI contract and model card. It is a prototype, not a claim of production readiness. Integration, load, monitoring, rollback, and current-data tests remain future work.
