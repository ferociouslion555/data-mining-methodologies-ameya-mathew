# SEMMA: Student Performance Classification

Dataset: UCI Student Performance dataset. The goal is to classify whether a student reaches a defined final-grade threshold using demographic, school, and study variables.

The notebook follows the SEMMA cycle:

1. **Sample:** obtain and create reproducible train/test partitions.
2. **Explore:** inspect distributions, missingness, class balance, and relationships.
3. **Modify:** encode categorical fields, scale numeric values where needed, and define the target before modeling.
4. **Model:** compare a baseline classifier with tree-based modeling.
5. **Assess:** report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and subgroup checks.

The target definition and threshold are documented in the notebook to avoid using the final grade as both target and predictor.

The executed Sample phase uses the UCI Mathematics course table (`student-mat.csv`), containing 395 rows and 33 source variables. The target is `pass = 1` when `G3 >= 10`, producing 265 passing and 130 non-passing records. A seeded stratified 80/20 split creates 316 training and 79 test rows, with a 0.6709 pass rate in both partitions.
