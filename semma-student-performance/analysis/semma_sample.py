from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'student-mat.csv', sep=';')
df['pass'] = (df['G3'] >= 10).astype(int)
train, test = train_test_split(df, test_size=.2, stratify=df['pass'], random_state=42)
out = ROOT / 'artifacts' / 'sample'; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'dataset':'UCI Student Performance', 'rows':[len(df)], 'columns':[len(df.columns)], 'target':'pass = G3 >= 10', 'positive_class_count':[int(df['pass'].sum())], 'negative_class_count':[int((df['pass']==0).sum())], 'train_rows':[len(train)], 'test_rows':[len(test)], 'random_seed':[42], 'sampling':'stratified 80/20 split'}).to_csv(out / 'sample_summary.csv', index=False)
pd.DataFrame({'split':['train','test'], 'rows':[len(train),len(test)], 'pass_rate':[train['pass'].mean(),test['pass'].mean()]}).to_csv(out / 'split_balance.csv', index=False)
print(pd.read_csv(out / 'sample_summary.csv').to_string(index=False))
print(pd.read_csv(out / 'split_balance.csv').to_string(index=False))
