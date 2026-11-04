from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'student-mat.csv', sep=';')
df['pass'] = (df['G3'] >= 10).astype(int)
out = ROOT / 'artifacts' / 'explore'; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'column':df.columns, 'dtype':[str(x) for x in df.dtypes], 'missing':[int(x) for x in df.isna().sum()], 'unique':[int(x) for x in df.nunique()]}).to_csv(out / 'data_dictionary.csv', index=False)
df.describe(include='all').transpose().to_csv(out / 'descriptive_statistics.csv')
df.groupby('pass').agg(students=('pass','size'), mean_final_grade=('G3','mean'), mean_studytime=('studytime','mean'), mean_absences=('absences','mean')).reset_index().to_csv(out / 'class_summary.csv', index=False)
numeric = df.select_dtypes('number')
numeric.corr()['pass'].sort_values(ascending=False).to_csv(out / 'correlations_with_target.csv')
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.countplot(data=df, x='pass', ax=axes[0], palette=['#E07A5F','#276FBF'], hue='pass', legend=False); axes[0].set_title('Target class balance'); axes[0].set_xlabel('Pass (0/1)')
sns.boxplot(data=df, x='pass', y='G3', ax=axes[1], hue='pass', palette=['#E07A5F','#276FBF'], legend=False); axes[1].set_title('Final grade by class')
sns.boxplot(data=df, x='pass', y='absences', ax=axes[2], hue='pass', palette=['#E07A5F','#276FBF'], legend=False); axes[2].set_title('Absences by class')
fig.tight_layout(); fig.savefig(out / 'fig_01_explore_classes.png', dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 5)); sns.heatmap(numeric.corr(), cmap='coolwarm', center=0, ax=ax); ax.set_title('Numeric correlation matrix'); fig.tight_layout(); fig.savefig(out / 'fig_02_correlation_matrix.png', dpi=160); plt.close(fig)
print('shape', df.shape, 'missing', int(df.isna().sum().sum()))
print(pd.read_csv(out / 'class_summary.csv').to_string(index=False))
print(pd.read_csv(out / 'correlations_with_target.csv').tail(8).to_string(index=False))
