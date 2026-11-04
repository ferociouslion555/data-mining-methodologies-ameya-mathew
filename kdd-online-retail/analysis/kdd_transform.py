from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
tx = pd.read_csv(ROOT / 'artifacts' / 'clean_transactions.csv', parse_dates=['InvoiceDate'])
out = ROOT / 'artifacts' / 'transformation'
out.mkdir(parents=True, exist_ok=True)
as_of = tx['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = tx.groupby('CustomerID').agg(recency_days=('InvoiceDate', lambda x: (as_of - x.max()).days), frequency=('InvoiceNo','nunique'), monetary=('Revenue','sum'), transactions=('InvoiceNo','nunique'), items=('Quantity','sum')).reset_index()
rfm.to_csv(out / 'customer_rfm.csv', index=False)
top_codes = tx['StockCode'].value_counts().head(100).index
subset = tx[tx['StockCode'].isin(top_codes)]
basket = pd.crosstab(subset['InvoiceNo'], subset['StockCode']).clip(upper=1)
basket.to_csv(out / 'basket_matrix_top100.csv')
pd.DataFrame({'metric':['as_of_date','customers','invoices','basket_rows','basket_products','rfm_columns'], 'value':[str(as_of.date()),len(rfm),tx['InvoiceNo'].nunique(),len(basket),len(basket.columns),len(rfm.columns)]}).to_csv(out / 'transformation_summary.csv', index=False)
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(14,4))
for ax, col, title in zip(axes, ['recency_days','frequency','monetary'], ['Recency','Frequency','Monetary']):
    ax.hist(rfm[col].clip(upper=rfm[col].quantile(.99)), bins=40, color='#276FBF')
    ax.set_title(title); ax.set_xlabel(col)
fig.tight_layout(); fig.savefig(out / 'fig_01_rfm_distributions.png', dpi=160); plt.close(fig)
print(rfm.describe().to_string())
print(pd.read_csv(out / 'transformation_summary.csv').to_string(index=False))
