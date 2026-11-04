from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
rfm = pd.read_csv(ROOT / 'artifacts' / 'transformation' / 'customer_rfm.csv')
out = ROOT / 'artifacts' / 'mining'; out.mkdir(parents=True, exist_ok=True)
features = ['recency_days','frequency','monetary']
X = StandardScaler().fit_transform(np.log1p(rfm[features]))
scores = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = model.fit_predict(X)
    scores.append({'k': k, 'silhouette': silhouette_score(X, labels), 'inertia': model.inertia_})
score_df = pd.DataFrame(scores); score_df.to_csv(out / 'cluster_selection.csv', index=False)
best_k = int(score_df.loc[score_df['silhouette'].idxmax(), 'k'])
model = KMeans(n_clusters=best_k, n_init=20, random_state=42)
rfm['cluster'] = model.fit_predict(X)
cluster_summary = rfm.groupby('cluster').agg(customers=('CustomerID','size'), recency_days=('recency_days','mean'), frequency=('frequency','mean'), monetary=('monetary','mean')).reset_index()
cluster_summary.to_csv(out / 'cluster_summary.csv', index=False)
rfm.to_csv(out / 'customer_clusters.csv', index=False)

basket = pd.read_csv(ROOT / 'artifacts' / 'transformation' / 'basket_matrix_top100.csv', index_col=0)
items = basket.columns.tolist(); n = len(basket)
rows = []
for i, item_a in enumerate(items):
    a = basket[item_a].to_numpy(dtype=bool); count_a = int(a.sum())
    if count_a == 0: continue
    for item_b in items[i+1:]:
        both = int((a & basket[item_b].to_numpy(dtype=bool)).sum())
        if both < 20: continue
        count_b = int(basket[item_b].sum())
        rows.append({'antecedent': item_a, 'consequent': item_b, 'support': both/n, 'confidence': both/count_a, 'lift': (both/n)/((count_a/n)*(count_b/n)), 'cooccurrence': both})
rules = pd.DataFrame(rows).sort_values(['lift','support'], ascending=False) if rows else pd.DataFrame(columns=['antecedent','consequent','support','confidence','lift','cooccurrence'])
rules.head(100).to_csv(out / 'association_rules_top100.csv', index=False)
print('cluster selection')
print(score_df.to_string(index=False))
print('cluster summary')
print(cluster_summary.to_string(index=False))
print('rules', len(rules), 'top rules')
print(rules.head(10).to_string(index=False))
