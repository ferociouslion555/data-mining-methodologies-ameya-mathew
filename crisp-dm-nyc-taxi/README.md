# CRISP-DM: NYC Taxi Trip Duration

## Objective

Estimate taxi trip duration from information available when an estimate is requested. The primary metric is RMSLE, supported by MAE and median absolute error in seconds.

## Six phases

- **Business understanding:** define ETA use, stakeholders, success criteria, and operational constraints.
- **Data understanding:** inspect target skew, time patterns, spatial density, invalid coordinates, and extreme durations.
- **Data preparation:** parse timestamps, apply documented quality rules, engineer distance/time features, and prevent target leakage.
- **Modeling:** compare a transparent linear baseline with histogram gradient boosting using a chronological holdout.
- **Evaluation:** report random versus future-month performance, segment errors, residual patterns, and limitations.
- **Deployment:** package prediction logic as a FastAPI prototype with an explicit input schema.

The prior analysis found a histogram gradient-boosting RMSLE of approximately 0.3334 on the June holdout, compared with approximately 0.4852 for linear regression. These results apply only to the declared sample and feature set; the project is not presented as production-ready.

## Files

- `notebooks/CRISP_DM_NYC_Taxi.ipynb`
- `reports/CRISP_DM_Report.md`
- `reports/hostile_review.md`
- `deployment/app.py`
- `deployment/model_card.md`
