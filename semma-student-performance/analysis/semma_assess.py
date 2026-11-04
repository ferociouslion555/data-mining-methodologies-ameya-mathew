from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'student-mat.csv', sep=';')
df['pass'] = (df['G3'] >= 10).astype(int)
excluded = ['G1', 'G2', 'G3', 'pass', 'Unnamed: 0']
features = [c for c in df.columns if c not in excluded]
numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
categorical = [c for c in features if c not in numeric]
pre = ColumnTransformer([('num', StandardScaler(), numeric), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)])
X_train, X_test, y_train, y_test = train_test_split(df[features], df['pass'], test_size=.2, stratify=df['pass'], random_state=42)
pipe = Pipeline([('preprocess', pre), ('model', RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight='balanced', random_state=42, n_jobs=-1))])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
prob = pipe.predict_proba(X_test)[:, 1]
out = ROOT / 'artifacts' / 'assess'
out.mkdir(parents=True, exist_ok=True)
(out / 'classification_report.txt').write_text(classification_report(y_test, pred, target_names=['not_pass','pass']))
pd.DataFrame(confusion_matrix(y_test, pred), index=['actual_not_pass','actual_pass'], columns=['predicted_not_pass','predicted_pass']).to_csv(out / 'confusion_matrix.csv')
(out / 'assessment_summary.txt').write_text(f'roc_auc={roc_auc_score(y_test, prob):.6f}\nfalse_positives={int(((pred==1)&(y_test==0)).sum())}\nfalse_negatives={int(((pred==0)&(y_test==1)).sum())}\n')
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['not pass','pass'], yticklabels=['not pass','pass'], ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Random forest confusion matrix')
fig.tight_layout(); fig.savefig(out / 'fig_03_confusion_matrix.png', dpi=160); plt.close(fig)
print((out / 'classification_report.txt').read_text()); print((out / 'assessment_summary.txt').read_text())
