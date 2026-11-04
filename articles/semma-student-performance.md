# Using SEMMA to Study Student Performance

SEMMA organizes this project around Sample, Explore, Modify, Model, and Assess. The UCI Mathematics student table contains 395 records. A pass target was defined as a final grade of at least 10, while final and prior-period grades were excluded from the early-warning predictor set.

A leakage-safe preprocessing pipeline encoded 30 predictors into 56 columns. A balanced random forest achieved ROC-AUC 0.6517 and passing-class recall 0.85 on the held-out test set. However, non-passing recall was only 0.35. The model is therefore an educational demonstration, not a system for admissions, discipline, or automated student labeling.
