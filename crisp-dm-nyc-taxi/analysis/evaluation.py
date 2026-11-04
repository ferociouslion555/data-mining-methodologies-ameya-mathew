from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'NYC_sample_200k.csv'
OUT = ROOT / 'artifacts' / 'evaluation'
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df = df[df['trip_duration'].between(60, 7200) & df['passenger_count'].between(1, 8)].copy()
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
df['pickup_month'] = df['pickup_datetime'].dt.month
df['hour_sin'] = np.sin(2*np.pi*df['pickup_hour']/24); df['hour_cos'] = np.cos(2*np.pi*df['pickup_hour']/24)
df['weekday_sin'] = np.sin(2*np.pi*df['pickup_weekday']/7); df['weekday_cos'] = np.cos(2*np.pi*df['pickup_weekday']/7)
lon1, lat1, lon2, lat2 = map(np.radians, [df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude']])
a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
df['haversine_km'] = 2*6371.0088*np.arcsin(np.sqrt(a)); df['store_and_fwd_flag'] = (df['store_and_fwd_flag']=='Y').astype(int)
features = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag','pickup_hour','pickup_weekday','pickup_month','hour_sin','hour_cos','weekday_sin','weekday_cos','haversine_km']
train = df[df['pickup_datetime'] < pd.Timestamp('2016-06-01')]; test = df[df['pickup_datetime'] >= pd.Timestamp('2016-06-01')].copy()
model = HistGradientBoostingRegressor(max_iter=250, learning_rate=.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
model.fit(train[features], np.log1p(train['trip_duration']))
test['prediction_seconds'] = np.maximum(0, np.expm1(model.predict(test[features])))
test['absolute_error_seconds'] = (test['trip_duration'] - test['prediction_seconds']).abs()
test['signed_error_seconds'] = test['prediction_seconds'] - test['trip_duration']
test['duration_band'] = pd.cut(test['trip_duration'], [0, 300, 600, 1200, 2400, 7200], labels=['1-5 min','5-10 min','10-20 min','20-40 min','40-120 min'])
test['distance_band'] = pd.cut(test['haversine_km'], [-.01, 1, 3, 7, 15, np.inf], labels=['0-1 km','1-3 km','3-7 km','7-15 km','15+ km'])

def summarize(group):
    pred = np.maximum(0, group['prediction_seconds'])
    return pd.Series({'rows': len(group), 'rmsle': np.sqrt(mean_squared_log_error(group['trip_duration'], pred)), 'mae_seconds': mean_absolute_error(group['trip_duration'], pred), 'median_absolute_error_seconds': group['absolute_error_seconds'].median(), 'mean_signed_error_seconds': group['signed_error_seconds'].mean()})

rows = []
for name, group in [('hour', test.groupby('pickup_hour', observed=False)), ('duration_band', test.groupby('duration_band', observed=False)), ('distance_band', test.groupby('distance_band', observed=False))]:
    part = group.apply(summarize, include_groups=False).reset_index(); part.insert(0, 'segment', name); rows.append(part)
segments = pd.concat(rows, ignore_index=True); segments.to_csv(OUT / 'segment_metrics.csv', index=False)
test[['id','trip_duration','prediction_seconds','absolute_error_seconds','signed_error_seconds','pickup_hour','haversine_km','duration_band','distance_band']].to_csv(OUT / 'june_predictions_and_residuals.csv', index=False)

hour = segments[segments['segment']=='hour']
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hour['pickup_hour'], hour['mae_seconds'], marker='o'); axes[0].set_title('June MAE by pickup hour'); axes[0].set_xlabel('Pickup hour'); axes[0].set_ylabel('MAE (seconds)')
axes[1].scatter(test['haversine_km'], test['signed_error_seconds'], s=5, alpha=.15); axes[1].axhline(0, color='black', lw=1); axes[1].set_xlim(0, test['haversine_km'].quantile(.99)); axes[1].set_title('June residuals versus distance'); axes[1].set_xlabel('Haversine distance (km)'); axes[1].set_ylabel('Prediction - actual (seconds)')
fig.tight_layout(); fig.savefig(OUT / 'fig_04_evaluation_segments.png', dpi=160); plt.close(fig)
print(segments.to_string(index=False))
