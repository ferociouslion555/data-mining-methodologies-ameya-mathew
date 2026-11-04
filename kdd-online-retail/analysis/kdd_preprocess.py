from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_excel(ROOT / 'data' / 'Online_Retail.xlsx')
raw['InvoiceDate'] = pd.to_datetime(raw['InvoiceDate'], errors='coerce')
audit = pd.DataFrame({'source_row': range(len(raw)), 'reason': 'retained'})
is_cancel = raw['InvoiceNo'].astype(str).str.startswith('C')
rules = [('cancelled_invoice', is_cancel), ('missing_customer_id', raw['CustomerID'].isna()), ('nonpositive_quantity', raw['Quantity'] <= 0), ('nonpositive_price', raw['UnitPrice'] <= 0), ('invalid_date', raw['InvoiceDate'].isna())]
for reason, mask in rules:
    audit.loc[(audit['reason'] == 'retained') & mask, 'reason'] = reason
audit.to_csv(ROOT / 'artifacts' / 'preprocessing_audit.csv', index=False)
clean = raw[audit['reason'].eq('retained')].copy()
clean['Revenue'] = clean['Quantity'] * clean['UnitPrice']
clean.to_csv(ROOT / 'artifacts' / 'clean_transactions.csv', index=False)
summary = audit['reason'].value_counts().rename_axis('reason').reset_index(name='rows')
summary['percent_of_raw'] = 100 * summary['rows'] / len(raw)
summary.to_csv(ROOT / 'artifacts' / 'preprocessing_summary.csv', index=False)
print(summary.to_string(index=False))
print('clean_rows', len(clean), 'clean_invoices', clean['InvoiceNo'].nunique(), 'clean_customers', clean['CustomerID'].nunique())
