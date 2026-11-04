from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_log_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'NYC_sample_200k.csv')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
df = df[df['trip_duration'].between(60, 7200) & df['passenger_count'].between(1, 8)].copy()
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
df['pickup_month'] = df['pickup_datetime'].dt.month
df['hour_sin'] = np.sin(2*np.pi*df['pickup_hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['pickup_hour']/24)
df['weekday_sin'] = np.sin(2*np.pi*df['pickup_weekday']/7)
df['weekday_cos'] = np.cos(2*np.pi*df['pickup_weekday']/7)
lon1, lat1, lon2, lat2 = map(np.radians, [df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude']])
a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
df['haversine_km'] = 2*6371.0088*np.arcsin(np.sqrt(a))
df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)

FEATURES = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag','pickup_hour','pickup_weekday','pickup_month','hour_sin','hour_cos','weekday_sin','weekday_cos','haversine_km']
X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df['trip_duration'].to_numpy()
y_log = np.log1p(y)

def score(y_true, pred_log):
    pred = np.maximum(0, np.expm1(pred_log))
    return {'rmsle': float(np.sqrt(mean_squared_log_error(y_true, pred))), 'mae_seconds': float(mean_absolute_error(y_true, pred)), 'median_absolute_error_seconds': float(median_absolute_error(y_true, pred)), 'r2_raw_seconds': float(r2_score(y_true, pred))}

def fit_score(name, model, train_idx, test_idx, split):
    model.fit(X.iloc[train_idx], y_log[train_idx])
    out = {'model': name, 'split': split, 'train_rows': len(train_idx), 'test_rows': len(test_idx)}
    out.update(score(y[test_idx], model.predict(X.iloc[test_idx])))
    return out

rng = np.random.RandomState(42)
idx = np.arange(len(df)); rng.shuffle(idx)
cut = int(.8 * len(idx))
random_train, random_test = idx[:cut], idx[cut:]
temporal_train = np.where(df['pickup_datetime'] < pd.Timestamp('2016-06-01'))[0]
temporal_test = np.where(df['pickup_datetime'] >= pd.Timestamp('2016-06-01'))[0]
models = {'ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)), 'hist_gradient_boosting': HistGradientBoostingRegressor(max_iter=250, learning_rate=.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)}
rows = []
for split, train, test in [('random_80_20', random_train, random_test), ('jan_may_to_june', temporal_train, temporal_test)]:
    for name, model in models.items():
        rows.append(fit_score(name, model, train, test, split))
out = ROOT / 'artifacts' / 'modeling'; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / 'model_comparison.csv', index=False)
(out / 'modeling_config.json').write_text(json.dumps({'features': FEATURES, 'random_seed': 42, 'target': 'log1p(trip_duration)', 'temporal_cutoff': '2016-06-01'}, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
