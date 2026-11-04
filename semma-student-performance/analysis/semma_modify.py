from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'student-mat.csv', sep=';')
df['pass'] = (df['G3'] >= 10).astype(int)
target = 'pass'
excluded = ['G1', 'G2', 'G3', 'pass', 'Unnamed: 0']
predictors = [c for c in df.columns if c not in excluded]
numeric = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
categorical = [c for c in predictors if c not in numeric]
preprocessor = ColumnTransformer([('numeric', StandardScaler(), numeric), ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical)])
pipeline = Pipeline([('preprocess', preprocessor)])
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=.2, stratify=df[target], random_state=42)
out = ROOT / 'artifacts' / 'modify'
out.mkdir(parents=True, exist_ok=True)
(out / 'modify_schema.json').write_text(json.dumps({'target': target, 'excluded_fields': excluded, 'predictors': predictors, 'numeric_features': numeric, 'categorical_features': categorical, 'random_seed': 42}, indent=2))
pd.DataFrame({'split':['train','test'], 'rows':[len(X_train),len(X_test)], 'pass_rate':[y_train.mean(),y_test.mean()]}).to_csv(out / 'modified_split.csv', index=False)
pipeline.fit(X_train, y_train)
transformed = pipeline.transform(X_train)
(out / 'transformed_shape.txt').write_text(f'train_rows={len(X_train)}\ntransformed_columns={transformed.shape[1]}\n')
print('predictors', len(predictors), 'numeric', len(numeric), 'categorical', len(categorical), 'transformed_columns', transformed.shape[1])
print('excluded', excluded)
