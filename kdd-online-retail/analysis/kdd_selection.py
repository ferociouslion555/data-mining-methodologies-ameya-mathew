from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_excel(ROOT / 'data' / 'Online_Retail.xlsx')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
out = ROOT / 'artifacts' / 'selection'
out.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame({'metric':['raw_rows','columns','date_min','date_max','unique_invoices','unique_stock_codes','unique_customers','missing_customer_ids','cancelled_invoice_rows','nonpositive_quantity_rows','nonpositive_price_rows'], 'value':[len(df),len(df.columns),str(df['InvoiceDate'].min()),str(df['InvoiceDate'].max()),df['InvoiceNo'].nunique(),df['StockCode'].nunique(),df['CustomerID'].nunique(),int(df['CustomerID'].isna().sum()),int(df['InvoiceNo'].astype(str).str.startswith('C').sum()),int((df['Quantity']<=0).sum()),int((df['UnitPrice']<=0).sum())]})
summary.to_csv(out / 'selection_summary.csv', index=False)
print(summary.to_string(index=False))
