from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'student-mat.csv', sep=';')
df['pass'] = (df['G3'] >= 10).astype(int)
excluded = ['G1', 'G2', 'G3', 'pass', 'Unnamed: 0']
predictors = [c for c in df.columns if c not in excluded]
numeric = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
categorical = [c for c in predictors if c not in numeric]
pre = ColumnTransformer([('num', StandardScaler(), numeric), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)])
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df['pass'], test_size=.2, stratify=df['pass'], random_state=42)
models = {'majority_baseline': DummyClassifier(strategy='most_frequent'), 'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42), 'random_forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight='balanced', random_state=42, n_jobs=-1)}
rows = []
for name, estimator in models.items():
    pipe = Pipeline([('preprocess', pre), ('model', estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    probability = pipe.predict_proba(X_test)[:, 1]
    rows.append({'model': name, 'accuracy': accuracy_score(y_test, pred), 'balanced_accuracy': balanced_accuracy_score(y_test, pred), 'precision': precision_score(y_test, pred, zero_division=0), 'recall': recall_score(y_test, pred, zero_division=0), 'f1': f1_score(y_test, pred, zero_division=0), 'roc_auc': roc_auc_score(y_test, probability)})
out = ROOT / 'artifacts' / 'model'
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / 'model_comparison.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
