# From Business Question to ETA Prototype: A CRISP-DM Taxi Project

This project applies CRISP-DM to NYC taxi trip-duration prediction. The work begins with an operational question—how long might a taxi trip take?—and proceeds through data understanding, auditable cleaning, leakage-safe feature engineering, modeling, evaluation, and a constrained deployment prototype.

The cleaned sample retained 198,523 of 200,000 rows. Histogram gradient boosting outperformed a Ridge baseline, reaching RMSLE 0.3508 and MAE 203.6 seconds on a June future holdout. Segment analysis revealed underprediction for long trips, so the model is not presented as production ready. The main lesson is that deployment is part of the data-science lifecycle, but honest limitations are part of deployment too.

Reproducibility scripts and evidence are in `crisp-dm-nyc-taxi/analysis/` and `crisp-dm-nyc-taxi/artifacts/`.
