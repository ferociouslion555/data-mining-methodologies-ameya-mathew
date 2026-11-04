"""Train and save the declared CRISP-DM candidate model."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'NYC_sample_200k.csv')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df = df[df['trip_duration'].between(60, 7200) & df['passenger_count'].between(1, 8)].copy()
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
df['pickup_month'] = df['pickup_datetime'].dt.month
df['hour_sin'] = np.sin(2*np.pi*df['pickup_hour']/24); df['hour_cos'] = np.cos(2*np.pi*df['pickup_hour']/24)
df['weekday_sin'] = np.sin(2*np.pi*df['pickup_weekday']/7); df['weekday_cos'] = np.cos(2*np.pi*df['pickup_weekday']/7)
lon1, lat1, lon2, lat2 = map(np.radians, [df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude']])
a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
df['haversine_km'] = 2*6371.0088*np.arcsin(np.sqrt(a))
df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
features = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag','pickup_hour','pickup_weekday','pickup_month','hour_sin','hour_cos','weekday_sin','weekday_cos','haversine_km']
train = df[df['pickup_datetime'] < pd.Timestamp('2016-06-01')]
model = HistGradientBoostingRegressor(max_iter=250, learning_rate=.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
model.fit(train[features], np.log1p(train['trip_duration']))
out = ROOT / 'deployment' / 'model.joblib'
joblib.dump({'model': model, 'features': features, 'target_transform': 'log1p'}, out)
print(f'saved {out}')
