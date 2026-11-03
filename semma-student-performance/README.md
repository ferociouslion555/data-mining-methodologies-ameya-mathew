# SEMMA: Student Performance Classification

Dataset: UCI Student Performance dataset. The goal is to classify whether a student reaches a defined final-grade threshold using demographic, school, and study variables.

The notebook follows the SEMMA cycle:

1. **Sample:** obtain and create reproducible train/test partitions.
2. **Explore:** inspect distributions, missingness, class balance, and relationships.
3. **Modify:** encode categorical fields, scale numeric values where needed, and define the target before modeling.
4. **Model:** compare a baseline classifier with tree-based modeling.
5. **Assess:** report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and subgroup checks.

The target definition and threshold are documented in the notebook to avoid using the final grade as both target and predictor.
