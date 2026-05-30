#!/usr/bin/env python
# V5: Maximum accuracy pipeline - optimized for Day 49 R² (=platform score)

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# ============ BASIC FEATURES ============
def parse_ts(ts):
    parts = ts.split(':')
    return int(parts[0]) * 60 + int(parts[1])

base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
b32_map = {c: i for i, c in enumerate(base32)}
gh_lat, gh_lon = {}, {}
for gh in set(train['geohash']) | set(test['geohash']):
    lat_lo, lat_hi = -90.0, 90.0; lon_lo, lon_hi = -180.0, 180.0; even = True
    for ch in gh:
        cd = base32.index(ch)
        for mask in [16, 8, 4, 2, 1]:
            if even:
                mid = (lon_lo + lon_hi) / 2
                if cd & mask: lon_lo = mid
                else: lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if cd & mask: lat_lo = mid
                else: lat_hi = mid
            even = not even
    gh_lat[gh] = (lat_lo + lat_hi) / 2
    gh_lon[gh] = (lon_lo + lon_hi) / 2

road_map = {'Residential': 0, 'Highway': 1, 'Street': 2}
weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}

le_gh = LabelEncoder()
le_gh.fit(pd.concat([train['geohash'], test['geohash']]))

for df in [train, test]:
    df['tmin'] = df['timestamp'].apply(parse_ts)
    df['hour'] = df['tmin'] // 60
    df['quarter'] = df['tmin'] // 15
    df['tod_bucket'] = df['tmin'] // 30
    df['min_sin'] = np.sin(2 * np.pi * df['tmin'] / 1440)
    df['min_cos'] = np.cos(2 * np.pi * df['tmin'] / 1440)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['lat'] = df['geohash'].map(gh_lat)
    df['lon'] = df['geohash'].map(gh_lon)
    df['LV'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['LM'] = (df['Landmarks'] == 'Yes').astype(int)
    df['road_enc'] = df['RoadType'].map(road_map).fillna(-1).astype(int)
    df['weather_enc'] = df['Weather'].map(weather_map).fillna(-1).astype(int)
    df['gh_enc'] = le_gh.transform(df['geohash'])
    df['temp_missing'] = df['Temperature'].isna().astype(int)
    df['road_missing'] = df['RoadType'].isna().astype(int)
    df['weather_missing'] = df['Weather'].isna().astype(int)
    for i in range(6):
        df[f'gh_c{i}'] = df['geohash'].str[i].map(b32_map)

# Temperature imputation
temp_by_gh = train.groupby('geohash')['Temperature'].median()
glob_temp = train['Temperature'].median()
for df in [train, test]:
    df['Temperature'] = df['Temperature'].fillna(df['geohash'].map(temp_by_gh)).fillna(glob_temp)

# ============ DAY 48 LOOKUP FEATURES ============
day48 = train[train['day'] == 48].copy()
day49 = train[train['day'] == 49].copy()
d48_demand = day48.set_index(['geohash', 'tmin'])['demand'].to_dict()
d48_gh_mean = day48.groupby('geohash')['demand'].mean().to_dict()
d48_global = day48['demand'].mean()

# Exact lookup + offsets
offsets = list(range(-180, 195, 15))  # -3h to +3h
offset_cols = []
for off in offsets:
    col = f'd48_t{off:+d}'
    offset_cols.append(col)
    d48_off = {(g, t + off): d for (g, t), d in d48_demand.items()}
    for df in [train, test]:
        keys = list(zip(df['geohash'], df['tmin']))
        df[col] = [d48_off.get(k, np.nan) for k in keys]
        df[col] = df[col].fillna(df['geohash'].map(d48_gh_mean)).fillna(d48_global)
    train.loc[train['day'] == 48, col] = np.nan
    train[col] = train[col].fillna(train['geohash'].map(d48_gh_mean)).fillna(d48_global)

print(f"Created {len(offset_cols)} temporal offset features")

# Window statistics
for df in [train, test]:
    odf = df[offset_cols]
    df['d48_w_mean'] = odf.mean(axis=1)
    df['d48_w_std'] = odf.std(axis=1)
    df['d48_w_min'] = odf.min(axis=1)
    df['d48_w_max'] = odf.max(axis=1)
    df['d48_w_range'] = df['d48_w_max'] - df['d48_w_min']
    df['d48_w_median'] = odf.median(axis=1)
    df['d48_slope'] = (df['d48_t+120'] - df['d48_t-120']) / 240
    df['d48_slope_s'] = (df['d48_t+30'] - df['d48_t-30']) / 60
    df['d48_curv'] = df['d48_t+15'] + df['d48_t-15'] - 2 * df['d48_t+0']
    df['d48_ratio'] = df['d48_t+0'] / (df['d48_w_mean'] + 1e-8)

window_cols = ['d48_w_mean', 'd48_w_std', 'd48_w_min', 'd48_w_max', 'd48_w_range',
               'd48_w_median', 'd48_slope', 'd48_slope_s', 'd48_curv', 'd48_ratio']

# Day 48 stats per geohash
d48_gh_agg = day48.groupby('geohash')['demand'].agg(['mean','std','median','min','max',
    lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)])
d48_gh_agg.columns = ['d48g_mean','d48g_std','d48g_med','d48g_min','d48g_max','d48g_q10','d48g_q90']
for c in d48_gh_agg.columns:
    for df in [train, test]:
        df[c] = df['geohash'].map(d48_gh_agg[c]).fillna(d48_global if 'std' not in c else 0)
d48g_cols = list(d48_gh_agg.columns)

# Day 48 per (geohash, hour)
d48_gh_hour = day48.groupby(['geohash', 'hour'])['demand'].agg(['mean','std','median']).reset_index()
d48_gh_hour.columns = ['geohash', 'hour', 'd48h_mean', 'd48h_std', 'd48h_med']
for df in [train, test]:
    m = df[['geohash','hour']].merge(d48_gh_hour, on=['geohash','hour'], how='left')
    for c in ['d48h_mean','d48h_std','d48h_med']:
        df[c] = m[c].values
        df[c] = df[c].fillna(df['geohash'].map(d48_gh_mean)).fillna(d48_global)
    if 'day' in df.columns:
        df.loc[df['day']==48, ['d48h_mean','d48h_std','d48h_med']] = np.nan
        for c in ['d48h_mean','d48h_std','d48h_med']:
            df[c] = df[c].fillna(df['geohash'].map(d48_gh_mean)).fillna(d48_global)
d48h_cols = ['d48h_mean','d48h_std','d48h_med']

# Day 48 per (geohash, quarter)
d48_gh_q = day48.groupby(['geohash', 'quarter'])['demand'].mean().reset_index()
d48_gh_q.columns = ['geohash', 'quarter', 'd48q_mean']
for df in [train, test]:
    m = df[['geohash','quarter']].merge(d48_gh_q, on=['geohash','quarter'], how='left')
    df['d48q_mean'] = m['d48q_mean'].values
    df['d48q_mean'] = df['d48q_mean'].fillna(df['geohash'].map(d48_gh_mean)).fillna(d48_global)
    if 'day' in df.columns:
        df.loc[df['day']==48, 'd48q_mean'] = np.nan
        df['d48q_mean'] = df['d48q_mean'].fillna(df['geohash'].map(d48_gh_mean)).fillna(d48_global)

# Day 48 per (geohash, feature_value) - conditional means
for feat_col, feat_name in [('RoadType','road'), ('NumberofLanes','lanes'),
                             ('Weather','weather'), ('LargeVehicles','lv'), ('Landmarks','lm')]:
    d48_cond = day48.groupby(['geohash', feat_col])['demand'].mean()
    for df in [train, test]:
        keys = list(zip(df['geohash'], df[feat_col]))
        df[f'd48c_{feat_name}'] = [d48_cond.get(k, np.nan) for k in keys]
        df[f'd48c_{feat_name}'] = df[f'd48c_{feat_name}'].fillna(df['geohash'].map(d48_gh_mean)).fillna(d48_global)
    train.loc[train['day']==48, f'd48c_{feat_name}'] = np.nan
    train[f'd48c_{feat_name}'] = train[f'd48c_{feat_name}'].fillna(train['geohash'].map(d48_gh_mean)).fillna(d48_global)

cond_cols = ['d48c_road', 'd48c_lanes', 'd48c_weather', 'd48c_lv', 'd48c_lm']

# Day 48 per (geohash, RoadType, hour) - triple conditional
d48_triple = day48.groupby(['geohash', 'RoadType', 'hour'])['demand'].mean()
for df in [train, test]:
    keys = list(zip(df['geohash'], df['RoadType'], df['hour']))
    df['d48c_road_hour'] = [d48_triple.get(k, np.nan) for k in keys]
    df['d48c_road_hour'] = df['d48c_road_hour'].fillna(df['d48c_road']).fillna(df['geohash'].map(d48_gh_mean)).fillna(d48_global)
train.loc[train['day']==48, 'd48c_road_hour'] = np.nan
train['d48c_road_hour'] = train['d48c_road_hour'].fillna(train['d48c_road']).fillna(train['geohash'].map(d48_gh_mean)).fillna(d48_global)

# Global road/weather means from day 48
d48_road_g = day48.groupby('RoadType')['demand'].mean().to_dict()
d48_weather_g = day48.groupby('Weather')['demand'].mean().to_dict()
for df in [train, test]:
    df['d48_road_g'] = df['RoadType'].map(d48_road_g).fillna(d48_global)
    df['d48_weather_g'] = df['Weather'].map(d48_weather_g).fillna(d48_global)

# ============ DAY 49 EARLY FEATURES ============
d49_gh = day49.groupby('geohash')['demand'].mean().to_dict()
d49_global = day49['demand'].mean()
for df in [train, test]:
    df['d49_gh'] = df['geohash'].map(d49_gh).fillna(d49_global)

# LOO for day 49
d49_sum = day49.groupby('geohash')['demand'].sum()
d49_cnt = day49.groupby('geohash')['demand'].count()
for idx in day49.index:
    gh = train.loc[idx, 'geohash']
    s, c = d49_sum.get(gh, 0), d49_cnt.get(gh, 0)
    train.loc[idx, 'd49_gh'] = (s - train.loc[idx, 'demand']) / (c - 1) if c > 1 else d49_global

# Day ratio
d48_early = day48[day48['tmin'] <= 120]
d48_early_m = d48_early.groupby('geohash')['demand'].mean().to_dict()
for df in [train, test]:
    df['d48_early_m'] = df['geohash'].map(d48_early_m).fillna(d48_global)
    df['day_ratio'] = df['d49_gh'] / (df['d48_early_m'] + 1e-8)
    df['demand_adj'] = df['d48_t+0'] * df['day_ratio']

# Day 49 conditional
d49_road = day49.groupby('RoadType')['demand'].mean().to_dict()
d49_gh_road = day49.groupby(['geohash', 'RoadType'])['demand'].mean()
for df in [train, test]:
    df['d49_road'] = df['RoadType'].map(d49_road).fillna(d49_global)
    keys = list(zip(df['geohash'], df['RoadType']))
    df['d49_gh_road'] = [d49_gh_road.get(k, np.nan) for k in keys]
    df['d49_gh_road'] = df['d49_gh_road'].fillna(df['d49_gh'])

print("Temporal features done")

# ============ INTERACTION FEATURES ============
for df in [train, test]:
    df['temp_x_hour'] = df['Temperature'] * df['hour']
    df['temp_x_road'] = df['Temperature'] * df['road_enc']
    df['temp_x_lanes'] = df['Temperature'] * df['NumberofLanes']
    df['road_x_lanes'] = df['road_enc'] * df['NumberofLanes']
    df['road_x_hour'] = df['road_enc'] * df['hour']
    df['lanes_x_hour'] = df['NumberofLanes'] * df['hour']
    df['lv_x_road'] = df['LV'] * df['road_enc']
    df['lm_x_road'] = df['LM'] * df['road_enc']
    df['lat_x_lon'] = df['lat'] * df['lon']
    df['temp_sq'] = df['Temperature'] ** 2
    df['hour_sq'] = df['hour'] ** 2
    df['tmin_x_road'] = df['tmin'] * df['road_enc']
    df['lv_x_lanes'] = df['LV'] * df['NumberofLanes']
    df['lm_x_lanes'] = df['LM'] * df['NumberofLanes']
    df['lv_x_lm'] = df['LV'] * df['LM']
    df['weather_x_road'] = df['weather_enc'] * df['road_enc']
    df['weather_x_lanes'] = df['weather_enc'] * df['NumberofLanes']
    df['weather_x_hour'] = df['weather_enc'] * df['hour']
    df['lat_x_hour'] = df['lat'] * df['hour']
    df['lon_x_hour'] = df['lon'] * df['hour']
    df['lat_x_tmin'] = df['lat'] * df['tmin']
    df['lon_x_tmin'] = df['lon'] * df['tmin']
    df['is_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 10) | (df['hour'] >= 16) & (df['hour'] <= 20)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['rush_x_lanes'] = df['is_rush'] * df['NumberofLanes']

interaction_cols = ['temp_x_hour', 'temp_x_road', 'temp_x_lanes', 'road_x_lanes',
    'road_x_hour', 'lanes_x_hour', 'lv_x_road', 'lm_x_road', 'lat_x_lon',
    'temp_sq', 'hour_sq', 'tmin_x_road', 'lv_x_lanes', 'lm_x_lanes', 'lv_x_lm',
    'weather_x_road', 'weather_x_lanes', 'weather_x_hour',
    'lat_x_hour', 'lon_x_hour', 'lat_x_tmin', 'lon_x_tmin',
    'is_rush', 'is_night', 'rush_x_lanes']

# Geohash frequency
gh_freq = train['geohash'].value_counts().to_dict()
for df in [train, test]:
    df['gh_freq'] = df['geohash'].map(gh_freq).fillna(0)

# ============ TARGET ENCODING (OOF) ============
GM = train['demand'].mean()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create composite keys
for df in [train, test]:
    df['gh_hour_k'] = df['geohash'] + '_' + df['hour'].astype(str)
    df['gh_quarter_k'] = df['geohash'] + '_' + df['quarter'].astype(str)
    df['gh_bucket_k'] = df['geohash'] + '_' + df['tod_bucket'].astype(str)
    df['gh_road_k'] = df['geohash'] + '_' + df['RoadType'].fillna('NA')
    df['gh_lanes_k'] = df['geohash'] + '_' + df['NumberofLanes'].astype(str)
    df['gh_weather_k'] = df['geohash'] + '_' + df['Weather'].fillna('NA')
    df['gh_lv_k'] = df['geohash'] + '_' + df['LV'].astype(str)
    df['gh_lm_k'] = df['geohash'] + '_' + df['LM'].astype(str)
    df['gh_road_hour_k'] = df['geohash'] + '_' + df['RoadType'].fillna('NA') + '_' + df['hour'].astype(str)
    df['gh_road_lanes_k'] = df['geohash'] + '_' + df['RoadType'].fillna('NA') + '_' + df['NumberofLanes'].astype(str)
    df['road_hour_k'] = df['RoadType'].fillna('NA') + '_' + df['hour'].astype(str)
    df['road_lanes_k'] = df['RoadType'].fillna('NA') + '_' + df['NumberofLanes'].astype(str)
    df['p5_hour_k'] = df['geohash'].str[:5] + '_' + df['hour'].astype(str)

te_configs = [
    ('geohash', 10), ('hour', 30), ('tmin', 25), ('quarter', 25),
    ('gh_hour_k', 5), ('gh_quarter_k', 3), ('gh_bucket_k', 4),
    ('gh_road_k', 5), ('gh_lanes_k', 5), ('gh_weather_k', 5),
    ('gh_lv_k', 5), ('gh_lm_k', 5),
    ('gh_road_hour_k', 2), ('gh_road_lanes_k', 2),
    ('road_hour_k', 15), ('road_lanes_k', 15), ('p5_hour_k', 8),
]

te_cols = []
for te_col, smooth in te_configs:
    col_name = f'te_{te_col}'
    te_cols.append(col_name)
    oof_te = np.zeros(len(train))
    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby(te_col)['demand'].agg(['count', 'mean'])
        sm = (agg['count'] * agg['mean'] + smooth * GM) / (agg['count'] + smooth)
        oof_te[val_idx] = train.iloc[val_idx][te_col].map(sm).fillna(GM).values
    train[col_name] = oof_te
    agg = train.groupby(te_col)['demand'].agg(['count', 'mean'])
    sm = (agg['count'] * agg['mean'] + smooth * GM) / (agg['count'] + smooth)
    test[col_name] = test[te_col].map(sm).fillna(GM).values

# Geohash aggregate stats via OOF
for agg_func, fill, feat_name in [
    ('std', 0, 'te_gh_std'), ('max', GM, 'te_gh_max'), ('min', 0, 'te_gh_min'),
    (lambda x: x.quantile(0.5), GM, 'te_gh_med'),
    (lambda x: x.quantile(0.25), GM, 'te_gh_q25'),
    (lambda x: x.quantile(0.75), GM, 'te_gh_q75'),
]:
    oof_a = np.zeros(len(train))
    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby('geohash')['demand'].agg(agg_func)
        if isinstance(agg, pd.Series): agg = agg.fillna(fill)
        oof_a[val_idx] = train.iloc[val_idx]['geohash'].map(agg).fillna(fill).values
    train[feat_name] = oof_a
    agg_full = train.groupby('geohash')['demand'].agg(agg_func)
    if isinstance(agg_full, pd.Series): agg_full = agg_full.fillna(fill)
    test[feat_name] = test['geohash'].map(agg_full).fillna(fill)

# Multi-key stats via OOF
for group_cols, agg_name, fill, feat in [
    (['geohash', 'hour'], 'median', GM, 'gh_hour_med'),
    (['geohash', 'hour'], 'std', 0, 'gh_hour_std'),
    (['geohash', 'tod_bucket'], 'median', GM, 'gh_bucket_med'),
]:
    oof_a = np.zeros(len(train))
    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby(group_cols)['demand'].agg(agg_name)
        lookup = train.iloc[val_idx].set_index(group_cols).index
        oof_a[val_idx] = [agg.get(k, fill) for k in lookup]
    train[feat] = oof_a
    agg_full = train.groupby(group_cols)['demand'].agg(agg_name)
    lookup_test = test.set_index(group_cols).index
    test[feat] = [agg_full.get(k, fill) for k in lookup_test]

# Range
oof_rng = np.zeros(len(train))
for tr_idx, val_idx in kf.split(train):
    rng = train.iloc[tr_idx].groupby('geohash')['demand'].agg(lambda x: x.max() - x.min())
    oof_rng[val_idx] = train.iloc[val_idx]['geohash'].map(rng).fillna(0).values
train['gh_range'] = oof_rng
test['gh_range'] = test['geohash'].map(train.groupby('geohash')['demand'].agg(lambda x: x.max() - x.min())).fillna(0)

agg_te_cols = ['te_gh_std', 'te_gh_max', 'te_gh_min', 'te_gh_med', 'te_gh_q25', 'te_gh_q75',
               'gh_hour_med', 'gh_hour_std', 'gh_bucket_med', 'gh_range']

print("Target encoding done")

# ============ FEATURE LIST ============
features = (
    ['gh_enc', 'day', 'tmin', 'hour', 'quarter', 'tod_bucket',
     'min_sin', 'min_cos', 'hour_sin', 'hour_cos',
     'lat', 'lon', 'NumberofLanes', 'Temperature',
     'LV', 'LM', 'road_enc', 'weather_enc',
     'temp_missing', 'road_missing', 'weather_missing', 'gh_freq']
    + [f'gh_c{i}' for i in range(6)]
    + offset_cols + window_cols + d48g_cols + d48h_cols + ['d48q_mean']
    + cond_cols + ['d48c_road_hour', 'd48_road_g', 'd48_weather_g']
    + ['d49_gh', 'd48_early_m', 'day_ratio', 'demand_adj', 'd49_road', 'd49_gh_road']
    + interaction_cols
    + te_cols + agg_te_cols
)

print(f"Total features: {len(features)}")

# ============ MODEL TRAINING ============
X_all = train[features]
y_all = train['demand']
y_log = np.log1p(y_all)

SEEDS = [42, 123, 2024]
N_SEEDS = len(SEEDS)

model_configs = {
    'lgb1': lambda seed: lgb.LGBMRegressor(
        n_estimators=20000, learning_rate=0.003,
        max_depth=13, num_leaves=600, min_child_samples=5,
        subsample=0.75, colsample_bytree=0.4,
        reg_alpha=0.1, reg_lambda=0.5,
        random_state=seed, n_jobs=-1, verbose=-1),
    'lgb2': lambda seed: lgb.LGBMRegressor(
        n_estimators=20000, learning_rate=0.005,
        max_depth=10, num_leaves=300, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.5,
        reg_alpha=0.3, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1),
    'lgb3': lambda seed: lgb.LGBMRegressor(
        n_estimators=20000, learning_rate=0.008,
        max_depth=7, num_leaves=100, min_child_samples=20,
        subsample=0.85, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=2.0,
        random_state=seed, n_jobs=-1, verbose=-1),
    'xgb1': lambda seed: xgb.XGBRegressor(
        n_estimators=20000, learning_rate=0.003,
        max_depth=12, min_child_weight=2,
        subsample=0.75, colsample_bytree=0.4, colsample_bylevel=0.6,
        reg_alpha=0.1, reg_lambda=0.5, gamma=0.02,
        random_state=seed, n_jobs=-1, early_stopping_rounds=500, eval_metric='rmse'),
    'xgb2': lambda seed: xgb.XGBRegressor(
        n_estimators=20000, learning_rate=0.005,
        max_depth=9, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.5, colsample_bylevel=0.7,
        reg_alpha=0.3, reg_lambda=1.5, gamma=0.05,
        random_state=seed, n_jobs=-1, early_stopping_rounds=500, eval_metric='rmse'),
    'cb1': lambda seed: CatBoostRegressor(
        iterations=20000, learning_rate=0.003,
        depth=11, l2_leaf_reg=0.5,
        subsample=0.75, random_seed=seed, verbose=0,
        early_stopping_rounds=500),
    'cb2': lambda seed: CatBoostRegressor(
        iterations=20000, learning_rate=0.005,
        depth=8, l2_leaf_reg=2.0,
        subsample=0.8, random_seed=seed, verbose=0,
        early_stopping_rounds=500),
    'et': lambda seed: ExtraTreesRegressor(
        n_estimators=1000, max_depth=35, min_samples_leaf=1,
        max_features=0.4, random_state=seed, n_jobs=-1),
}

model_names = list(model_configs.keys())
oof_models = {k: np.zeros(len(train)) for k in model_names}
test_models = {k: np.zeros(len(test)) for k in model_names}

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*60}\nSEED {SEED} ({seed_idx+1}/{N_SEEDS})\n{'='*60}")
    kf_seed = KFold(n_splits=5, shuffle=True, random_state=SEED)

    _oof = {k: np.zeros(len(train)) for k in model_names}
    _test = {k: np.zeros(len(test)) for k in model_names}

    for fold, (tr_idx, val_idx) in enumerate(kf_seed.split(X_all)):
        print(f"  Fold {fold}...", end=" ", flush=True)
        X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]

        for name in model_names:
            m = model_configs[name](SEED)

            if name.startswith('lgb'):
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(500), lgb.log_evaluation(0)])
            elif name.startswith('xgb'):
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif name.startswith('cb'):
                m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            else:
                m.fit(X_tr, y_tr)

            _oof[name][val_idx] = m.predict(X_val)
            _test[name] += m.predict(test[features]) / 5

        # Print scores
        scores = {k: r2_score(y_val, _oof[k][val_idx]) for k in model_names}
        d49_val = [i for i in val_idx if train.iloc[i]['day'] == 49]
        d49_scores = {k: r2_score(y_all.iloc[d49_val], _oof[k][d49_val]) for k in model_names} if d49_val else {}
        best = max(scores, key=scores.get)
        print(f"Best={best.upper()}:{scores[best]:.4f}", end="")
        if d49_scores:
            best49 = max(d49_scores, key=d49_scores.get)
            print(f" D49_best={best49.upper()}:{d49_scores[best49]:.4f}")
        else:
            print()

    for k in model_names:
        oof_models[k] += _oof[k] / N_SEEDS
        test_models[k] += _test[k] / N_SEEDS

# ============ ENSEMBLE ============
y_true = train['demand'].values
d49_mask = (train['day'] == 49).values

print(f"\n{'='*60}\nMODEL SCORES\n{'='*60}")
for k in model_names:
    p = np.clip(oof_models[k], 1e-7, 1.0)
    r2_all = r2_score(y_true, p)
    r2_49 = r2_score(y_true[d49_mask], p[d49_mask])
    print(f"{k.upper():6s}: overall={r2_all:.6f}, day49={r2_49:.6f}")

# Scipy optimize on Day 49 R²
preds_list = [np.clip(oof_models[k], 1e-7, 1.0) for k in model_names]
test_list = [np.clip(test_models[k], 1e-7, 1.0) for k in model_names]

def neg_r2_d49(w):
    w = np.abs(w)
    w = w / w.sum()
    p = sum(w[i] * preds_list[i] for i in range(len(model_names)))
    return -r2_score(y_true[d49_mask], p[d49_mask])

def neg_r2_all(w):
    w = np.abs(w)
    w = w / w.sum()
    p = sum(w[i] * preds_list[i] for i in range(len(model_names)))
    return -r2_score(y_true, p)

# Optimize for Day 49 (platform score)
n = len(model_names)
res49 = minimize(neg_r2_d49, [1/n]*n, method='Nelder-Mead',
                 options={'maxiter': 200000, 'xatol': 1e-12, 'fatol': 1e-12})
w49 = np.abs(res49.x); w49 = w49 / w49.sum()
d49_r2 = -res49.fun

# Also optimize for overall
res_all = minimize(neg_r2_all, [1/n]*n, method='Nelder-Mead',
                   options={'maxiter': 200000, 'xatol': 1e-12, 'fatol': 1e-12})
w_all = np.abs(res_all.x); w_all = w_all / w_all.sum()

# Ridge stacking
oof_stack = np.column_stack(preds_list)
test_stack = np.column_stack(test_list)

meta_oof = np.zeros(len(train))
meta_test = np.zeros(len(test))
kf_meta = KFold(n_splits=10, shuffle=True, random_state=42)
for tr_idx, val_idx in kf_meta.split(oof_stack):
    meta = Ridge(alpha=0.1)
    meta.fit(oof_stack[tr_idx], y_true[tr_idx])
    meta_oof[val_idx] = meta.predict(oof_stack[val_idx])
    meta_test += meta.predict(test_stack) / 10

stacked_r2_49 = r2_score(y_true[d49_mask], np.clip(meta_oof[d49_mask], 1e-7, 1.0))

# Blend: stack + optimized
opt_oof_49 = sum(w49[i] * preds_list[i] for i in range(n))
blend_oof = 0.5 * np.clip(meta_oof, 1e-7, 1.0) + 0.5 * opt_oof_49
blend_r2_49 = r2_score(y_true[d49_mask], np.clip(blend_oof[d49_mask], 1e-7, 1.0))

print(f"\n{'='*60}\nENSEMBLE RESULTS (Day 49 = Platform Score)\n{'='*60}")
print(f"Optimized weights (D49): {dict(zip([k.upper() for k in model_names], [f'{x:.4f}' for x in w49]))}")
print(f"Day49 Optimized R2:  {d49_r2:.6f} (score: {100*d49_r2:.2f})")
print(f"Day49 Stacking R2:   {stacked_r2_49:.6f} (score: {100*stacked_r2_49:.2f})")
print(f"Day49 Blend R2:      {blend_r2_49:.6f} (score: {100*blend_r2_49:.2f})")

# Overall scores
opt_all_oof = sum(w_all[i] * preds_list[i] for i in range(n))
print(f"\nOverall Optimized R2: {-res_all.fun:.6f}")
print(f"Overall Stacking R2:  {r2_score(y_true, np.clip(meta_oof, 1e-7, 1.0)):.6f}")

# Choose best method based on Day 49 R²
methods_49 = {'optimized_d49': d49_r2, 'stacking': stacked_r2_49, 'blend': blend_r2_49}
best_method = max(methods_49, key=methods_49.get)
best_score = methods_49[best_method]
print(f"\n>>> Best method: {best_method}")
print(f">>> Platform Score estimate: {100*best_score:.2f}")

# Generate submission
if best_method == 'stacking':
    test['demand'] = np.clip(meta_test, 1e-7, 1.0)
elif best_method == 'optimized_d49':
    test['demand'] = np.clip(sum(w49[i] * test_list[i] for i in range(n)), 1e-7, 1.0)
else:
    test_opt = sum(w49[i] * test_list[i] for i in range(n))
    test['demand'] = np.clip(0.5 * meta_test + 0.5 * test_opt, 1e-7, 1.0)

submission = test[['Index', 'demand']]
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission shape: {submission.shape}")
print(f"Demand range: [{submission['demand'].min():.6f}, {submission['demand'].max():.6f}]")
print("Submission saved to submission.csv")
