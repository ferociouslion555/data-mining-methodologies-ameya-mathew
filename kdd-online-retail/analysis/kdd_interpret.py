from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'artifacts' / 'mining'
tx = pd.read_csv(ROOT / 'artifacts' / 'clean_transactions.csv')
rules = pd.read_csv(out / 'association_rules_top100.csv')
desc = tx[['StockCode','Description']].dropna().drop_duplicates('StockCode')
desc['StockCode'] = desc['StockCode'].astype(str)
for col in ['antecedent','consequent']:
    rules[col] = rules[col].astype(str)
    rules = rules.merge(desc.rename(columns={'StockCode': col, 'Description': col+'_description'}), on=col, how='left')
rules.to_csv(out / 'association_rules_interpreted.csv', index=False)
clusters = pd.read_csv(out / 'customer_clusters.csv')
sns.set_theme(style='whitegrid')
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=clusters, x='recency_days', y='monetary', hue='cluster', size='frequency', sizes=(20,180), alpha=.65, ax=ax)
ax.set_xlim(0, clusters['recency_days'].quantile(.99)); ax.set_ylim(0, clusters['monetary'].quantile(.99)); ax.set_title('Customer clusters in RFM space')
fig.tight_layout(); fig.savefig(out / 'fig_02_customer_clusters.png', dpi=160); plt.close(fig)
print(rules.head(10).to_string(index=False))
