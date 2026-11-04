# CRISP-DM Report: NYC Taxi Trip Duration

Student: Ameya Mathew  
SJSU ID: 019110936

## Project definition

This project applies the complete CRISP-DM lifecycle to a taxi-trip-duration prediction problem. The work is organized around a real operational question rather than only a model-fitting exercise: can a taxi platform estimate trip duration accurately enough to support passenger expectations and dispatch planning?

## Business understanding

### Business problem

Taxi passengers and dispatch operators need an estimate of how long a trip will take. Underestimates create missed expectations and planning errors; overestimates reduce confidence and can distort capacity planning. The proposed analytical solution predicts trip duration in seconds from trip information available when the estimate is requested.

### Stakeholders and decisions

| Stakeholder | Decision supported | Required output |
|---|---|---|
| Passenger | Whether the expected arrival time is acceptable | ETA in minutes/seconds |
| Dispatcher | Which vehicle or service option to prioritize | Comparable trip-duration estimate |
| Operations analyst | Whether model performance is improving | Time- and segment-level error report |
| Model owner | Whether to deploy or retrain | Evaluation, drift, and limitation evidence |

### Data-mining objective

Build and evaluate a supervised regression model that predicts `trip_duration` using pickup time, passenger count, and valid spatial or trip-context features. The model must exclude any feature that is only known after the trip has ended, especially `dropoff_datetime` and the target itself.

### Success criteria

| Dimension | Criterion |
|---|---|
| Modeling | Beat a transparent linear-regression baseline on a held-out future period |
| Primary metric | Minimize RMSLE because proportional ETA error matters more than raw error alone |
| Operational metric | Report MAE and median absolute error in seconds/minutes |
| Generalization | Compare random validation with chronological future-month validation |
| Trustworthiness | Report data-quality exclusions, leakage checks, segment errors, and unsupported claims |
| Deployment | Provide a reproducible prediction contract and model card; do not claim production readiness without integration testing |

### Constraints and risks

- A historical sample may not represent current taxi operations.
- Coordinates may contain defaults, outliers, or collection errors.
- Destination-based features are valid only when the destination is known before prediction.
- Random splits can be optimistic when recurring locations and time patterns appear in both partitions.
- The dataset does not contain all important causes of duration, such as traffic, weather, road closures, and events.
- A point prediction without an uncertainty interval is incomplete for operational use.

### Project plan

The next phases inspect the source data, define and audit cleaning rules, build leakage-safe features, compare baseline models, evaluate future-month and segment performance, and package a limited deployment prototype. Every claim in the final report is restricted to the data and experiments actually executed.

## Data understanding

### Dataset and variables

The project uses the NYC Taxi Trip Duration dataset. The analytical table contains a trip identifier, vendor identifier, pickup timestamp, drop-off timestamp, passenger count, pickup coordinates, drop-off coordinates, and the target trip duration in seconds. Each row represents one completed trip.

### Initial questions

1. Are timestamps parseable and within the expected study period?
2. Is the target heavily skewed or dominated by implausible extremes?
3. Are passenger counts, coordinates, and durations within defensible ranges?
4. Do trip duration and trip volume vary by hour, weekday, and month?
5. Are pickup and drop-off points concentrated in plausible NYC operating areas?
6. Which apparent patterns could be collection artifacts rather than useful predictive signal?

### Required exploratory evidence

The notebook should produce a data dictionary, row and column counts, missing-value counts, duplicate counts, summary statistics, a target histogram with a log-scale comparison, hourly and weekday summaries, coordinate-density plots, and an outlier audit. Each plot must state whether it is descriptive only or motivates a preparation decision.

### Findings from the executed analysis

The executed data-understanding script found 200,000 rows and 11 raw columns. After adding analysis-only time fields, the working table had 14 columns. There were no missing cells, duplicate rows, duplicate trip IDs, or invalid pickup/drop-off timestamps. The data covers January 1 through June 30, 2016. The target had a minimum of 1 second, a median of 663 seconds (11.05 minutes), a mean of 954.27 seconds (15.90 minutes), and a maximum of 86,390 seconds.

Quality flags found 311 trips at or above two hours, 256 trips at or above 80,000 seconds, 10 records with zero passengers, and 26 pickup coordinates outside the broad display window of longitude -74.30 to -73.65 and latitude 40.45 to 40.95. These records are not automatically declared erroneous. They are retained in the audit and handled through explicit preparation rules and sensitivity checks. The target is strongly right-skewed, so a log-scale target view and RMSLE are appropriate. The hourly and spatial plots motivate time and distance features, but they are descriptive and do not prove causation.

### Data-understanding decision

The next phase will preserve an audit of every excluded or flagged row. The model-development population will be defined by documented rules before model comparison, and sensitivity checks will be used where a rule could materially change the result.

## Data preparation

The preparation script parses timestamps, validates duration and passenger-count domains, checks coordinate finiteness, calculates haversine distance, encodes cyclic hour and weekday variables, and keeps the raw target separate from the predictor allowlist. The cleaning decisions are recorded row by row in `artifacts/data_preparation/exclusion_audit.csv`.

The fixed development rules retained durations from 60 through 7,200 seconds, passenger counts from 1 through 8, valid timestamps, and finite coordinates. These rules excluded 1,473 records for duration and 4 records for passenger count, leaving 198,523 of 200,000 rows. The rules are population restrictions selected before model comparison; they are not claims that every excluded record is impossible in the real world.

The engineered features include haversine distance, hour/month/weekday, and cyclic sine/cosine representations. `dropoff_datetime`, `trip_duration`, and `log_trip_duration` are explicitly excluded from predictors. The generated predictor allowlist has no overlap with the target or target-derived fields, providing a direct leakage check.

Large derived tables are local evidence artifacts and are not required to be committed to GitHub. The script is the reproducible source for regenerating them from the raw CSV.

## Modeling

The modeling script compares a scaled Ridge regression baseline with `HistGradientBoostingRegressor`. Both models predict `log1p(trip_duration)` and predictions are transformed back with `expm1` and clipped at zero before scoring. The same fixed feature allowlist and random seed are used for both models.

The random split uses 80% training and 20% testing. The temporal split trains on January through May and evaluates on June. The temporal split is the more realistic primary protocol because the intended use is future trip estimation.

The executed comparison produced the following results:

| Model | Split | RMSLE | MAE seconds | Median absolute error seconds |
|---|---|---:|---:|---:|
| Ridge | Random 80/20 | 0.5385 | 4,882.7 | 209.4 |
| Histogram gradient boosting | Random 80/20 | 0.3419 | 187.9 | 119.9 |
| Ridge | January–May → June | 0.5412 | 397.1 | 219.6 |
| Histogram gradient boosting | January–May → June | 0.3508 | 203.6 | 127.7 |

The boosting model is the selected candidate because it outperforms the baseline under both protocols. The June result is weaker than the random result, which is consistent with random-split optimism and is why the future-month result is reported separately.

## Evaluation

The selected gradient-boosting model was refit on January–May and evaluated on the June holdout. Overall performance was RMSLE 0.3508, MAE 203.6 seconds, and median absolute error 127.7 seconds. Segment analysis shows that performance is not uniform. The model had its largest RMSLE for 1–5 minute trips (0.4958) and 40–120 minute trips (0.4606). It also underpredicted the longest trips: the mean signed error for the 40–120 minute band was -670.3 seconds. Hour-level MAE was highest around 16:00–17:00, reaching about 262.8 and 257.8 seconds respectively.

Distance segments show lower RMSLE for longer trips but larger absolute errors in seconds. This illustrates why both proportional and operational metrics are needed. The segment results are diagnostic, not evidence of causation or fairness. They identify where the model should not be trusted without additional data and validation.

The generated June residual table and segment metrics are saved under `artifacts/evaluation/`, along with a residual plot. The final conclusion is narrow: on this cleaned sample and declared feature set, histogram gradient boosting outperformed the Ridge baseline under both random and chronological evaluation, but substantial segment errors remain.

## Deployment

The candidate model was serialized to `deployment/model.joblib` with its feature allowlist and `log1p` target-transform declaration. `deployment/app.py` defines `/health` and `/predict`, validates request fields with Pydantic, reconstructs the same engineered features, and returns an estimated duration in seconds and minutes. Direct model loading and prediction were verified locally with a representative request, producing a prediction of approximately 1,046 seconds (17.44 minutes).

The FastAPI HTTP integration was not executed in this environment because the FastAPI package was unavailable. Therefore the service is documented as a deployment prototype, not as a verified production service. HTTP integration, malformed-input tests, load tests, latency measurements, monitoring, security review, model rollback, and current-data validation remain required before operational use.
