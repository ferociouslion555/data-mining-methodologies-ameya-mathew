# SEMMA Report

This project uses the UCI Student Performance data to demonstrate the five SEMMA stages. Sampling is separated from exploration; modification is performed inside a scikit-learn pipeline; models are compared against a majority-class baseline; and assessment includes class-imbalance-aware metrics and a limitations section.

## Sample

The Mathematics-course table contains 395 students and 33 source variables. The target is defined before model fitting as `pass = 1` when the final grade `G3` is at least 10. This creates 265 positive and 130 negative cases. A stratified 80/20 split with random seed 42 produces 316 training and 79 test rows with identical pass rates of 0.6709. For an early-warning interpretation, `G1` and `G2` will be excluded from the final predictor set because those grades may not be available at the intended intervention time.

## Explore

The executed exploration found no missing cells in the 395-row table. The non-passing group had a mean final grade of 5.38, while the passing group had a mean of 12.88. The non-passing group had a higher mean absence count (6.76 versus 5.19), although this is descriptive and not causal. `failures` had the strongest negative non-target correlation among the inspected numeric variables (approximately -0.338), while age and going out showed modest negative relationships.

The final grade `G3` is used to construct the target and must never be a predictor. `G1` and `G2` are also excluded for the early-warning scenario. Correlations guide exploration only; they do not establish feature importance or causation.

## Modify

The modification stage excludes `G1`, `G2`, `G3`, the derived target `pass`, and the unnamed index field. The resulting allowlist contains 30 predictors: 13 numeric and 17 categorical. Numeric variables are standardized and categorical variables are one-hot encoded inside a scikit-learn `ColumnTransformer`, fitted only on the training partition. The transformed training matrix contains 56 columns. This pipeline prevents test-set statistics and unseen-category handling from leaking into training.

## Model

Three classifiers were trained on the same stratified split: a majority-class baseline, balanced logistic regression, and a balanced random forest. The random forest achieved the strongest test ROC-AUC (0.6517), balanced accuracy (0.5976), recall (0.8491), and F1 (0.7826). The majority baseline has accuracy 0.6709 but balanced accuracy 0.5000 and ROC-AUC 0.5000, showing why accuracy alone is inadequate for this imbalanced target. The results are educational and should not be interpreted as a validated student-risk intervention model.

## Assess

The random-forest confusion matrix contains 17 false positives and 8 false negatives on the 79-row test set. Recall for the passing class is 0.85, while recall for the non-passing class is only 0.35. A support or intervention system would need to decide whether false positives or false negatives are more costly; this notebook does not make that policy decision. The test set is small, the dataset is historical, and no external or temporal validation was performed. The model must not be used for admissions, discipline, or automated labeling of students.

The result is an educational classifier, not a high-stakes admissions or disciplinary tool. It should not be used to label individual students without consent, context, and human review.
