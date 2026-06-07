# %% [markdown]
# # Notebook 1: Data Preprocessing
# **EV Dynamic Tariff Optimization -- Agentic AI Framework**
#
# This notebook handles loading, cleaning, and feature engineering for both datasets:
# - **UrbanEV (ST-EVCDP):** Shenzhen charging pile data (247 zones, 30 days, 5-min intervals)
# - **ACN-Data:** Caltech/JPL individual charging sessions (~16K sessions)
#
# The processed data is saved to `data/processed/` for use in subsequent notebooks.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# paths
def get_project_root():
    """Find the project root whether this runs as a .py file or notebook."""
    start_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    current = os.path.abspath(start_dir)
    for _ in range(5):
        if os.path.isdir(os.path.join(current, 'data')) and os.path.isdir(os.path.join(current, 'notebooks')):
            return current
        current = os.path.dirname(current)
    return os.path.abspath(os.path.join(os.getcwd(), '..'))

BASE_DIR = get_project_root()
URBAN_DIR = os.path.join(BASE_DIR, 'data', 'urbanev')
ACN_DIR = os.path.join(BASE_DIR, 'data', 'acn')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"UrbanEV data: {URBAN_DIR}")
print(f"ACN data: {ACN_DIR}")
print(f"Processed output: {PROCESSED_DIR}")

# %% [markdown]
# ---
# ## 2. Load UrbanEV Dataset
# The UrbanEV dataset has **8,640 timestamps** (5-minute intervals over 30 days) across **247 traffic zones** in Shenzhen.
# Each CSV has the same structure: `timestamp` column (1-8640) + 247 zone columns.

# %%
# load all urbanev csvs
time_df = pd.read_csv(os.path.join(URBAN_DIR, 'time.csv'))
occupancy_df = pd.read_csv(os.path.join(URBAN_DIR, 'occupancy.csv'))
volume_df = pd.read_csv(os.path.join(URBAN_DIR, 'volume.csv'))
duration_df = pd.read_csv(os.path.join(URBAN_DIR, 'duration.csv'))
price_df = pd.read_csv(os.path.join(URBAN_DIR, 'price.csv'))
stations_df = pd.read_csv(os.path.join(URBAN_DIR, 'stations.csv'))
info_df = pd.read_csv(os.path.join(URBAN_DIR, 'information.csv'))
adj_df = pd.read_csv(os.path.join(URBAN_DIR, 'adj.csv'))

print("=== UrbanEV Data Shapes ===")
for name, df in [('time', time_df), ('occupancy', occupancy_df), ('volume', volume_df),
                 ('duration', duration_df), ('price', price_df), ('stations', stations_df),
                 ('information', info_df), ('adjacency', adj_df)]:
    print(f"  {name:>12s}: {str(df.shape):>14s}")

# %%
# quick look at the time index - need to construct actual datetime from month/day/year/hour/minute
print("Time dataframe sample:")
print(time_df.head())

# %% [markdown]
# ### 2.1 Construct Proper Datetime Index
# The `time.csv` has separate month, day, year, hour, minute, second columns.
# We'll combine them into a proper datetime column and use it as the index.

# %%
# build datetime column from the component columns
time_df['datetime'] = pd.to_datetime(
    time_df[['year', 'month', 'day', 'hour', 'minute', 'second']]
)
print(f"Date range: {time_df['datetime'].min()} to {time_df['datetime'].max()}")
print(f"Total timestamps: {len(time_df)}")
print(f"Time interval: {(time_df['datetime'].iloc[1] - time_df['datetime'].iloc[0]).total_seconds() / 60} minutes")

# %% [markdown]
# ### 2.2 Merge Datasets into a Unified Long-Format DataFrame
# Currently each file is in **wide format** (zones as columns). We'll convert to **long format**
# for easier analysis: one row per (timestamp, zone) combination.

# %%
# get zone column names (everything except 'timestamp')
zone_cols = [c for c in occupancy_df.columns if c != 'timestamp']
print(f"Number of zones: {len(zone_cols)}")
print(f"Zone IDs (first 10): {zone_cols[:10]}")

# %%
# melt each dataframe from wide to long format
def melt_urban(df, value_name):
    """Convert wide-format zone data to long format."""
    melted = df.melt(id_vars='timestamp', var_name='zone_id', value_name=value_name)
    melted['zone_id'] = melted['zone_id'].astype(str)
    return melted

occ_long = melt_urban(occupancy_df, 'occupancy')
vol_long = melt_urban(volume_df, 'volume_kwh')
dur_long = melt_urban(duration_df, 'duration_hrs')
price_long = melt_urban(price_df, 'price_cny_kwh')

print(f"Long format rows: {len(occ_long):,} (should be {8640 * 247:,})")

# %%
# merge all into a single dataframe
urban_long = occ_long.copy()
urban_long = urban_long.merge(vol_long, on=['timestamp', 'zone_id'], how='left')
urban_long = urban_long.merge(dur_long, on=['timestamp', 'zone_id'], how='left')
urban_long = urban_long.merge(price_long, on=['timestamp', 'zone_id'], how='left')

# add datetime from time_df
time_map = time_df[['datetime']].copy()
time_map.index = range(1, len(time_map) + 1)
urban_long['datetime'] = urban_long['timestamp'].map(time_map['datetime'])

print(f"Merged UrbanEV shape: {urban_long.shape}")
print(urban_long.head())

# %% [markdown]
# ### 2.3 Add Zone Metadata
# The `information.csv` contains per-zone metadata: total charging piles, fast/slow counts,
# area, whether it's in the CBD, and whether it has dynamic pricing.

# %%
print("Zone information columns:", list(info_df.columns))
print(f"\nZones with dynamic pricing: {info_df['dynamic_pricing'].sum()} out of {len(info_df)}")
print(f"CBD zones: {info_df['CBD'].sum()}")
print(f"\nTotal charging piles: {info_df['count'].sum()}")
print(f"Fast piles: {info_df['fast_count'].sum()}")
print(f"Slow piles: {info_df['slow_count'].sum()}")

# %%
# merge zone info into the long dataframe
# the 'grid' column in info_df corresponds to zone_id
info_merge = info_df[['grid', 'count', 'fast_count', 'slow_count', 'area', 'CBD', 'dynamic_pricing']].copy()
info_merge['grid'] = info_merge['grid'].astype(str)
info_merge = info_merge.rename(columns={
    'grid': 'zone_id',
    'count': 'total_piles',
    'fast_count': 'fast_piles',
    'slow_count': 'slow_piles'
})

urban_long = urban_long.merge(info_merge, on='zone_id', how='left')
print(f"Shape after zone info merge: {urban_long.shape}")

# %% [markdown]
# ### 2.4 Feature Engineering -- UrbanEV
# We create features that capture temporal patterns, utilization, and demand intensity.

# %%
# --- temporal features ---
urban_long['hour'] = urban_long['datetime'].dt.hour
urban_long['day_of_week'] = urban_long['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
urban_long['is_weekend'] = (urban_long['day_of_week'] >= 5).astype(int)
urban_long['day_name'] = urban_long['datetime'].dt.day_name()
urban_long['date'] = urban_long['datetime'].dt.date

# time period classification
def classify_period(hour):
    """Classify hour into peak, shoulder, or off-peak period."""
    if hour in range(8, 12) or hour in range(17, 21):  # 8-11am, 5-8pm
        return 'peak'
    elif hour in range(23, 24) or hour in range(0, 6):  # 11pm-5am
        return 'off_peak'
    else:
        return 'shoulder'

urban_long['time_period'] = urban_long['hour'].apply(classify_period)

# %%
# --- utilization rate ---
# occupancy = number of piles currently occupied
# utilization_rate = occupancy / total_piles in that zone
urban_long['utilization_rate'] = urban_long['occupancy'] / urban_long['total_piles']
urban_long['utilization_rate'] = urban_long['utilization_rate'].clip(0, 1)  # cap at 100%

# %%
# --- demand intensity features ---
# revenue proxy: volume * price
urban_long['revenue_proxy'] = urban_long['volume_kwh'] * urban_long['price_cny_kwh']

# energy per pile (intensity measure)
urban_long['kwh_per_pile'] = urban_long['volume_kwh'] / urban_long['total_piles']

# queue length proxy: if occupancy > total piles, we assume a queue
urban_long['queue_proxy'] = np.maximum(0, urban_long['occupancy'] - urban_long['total_piles'])

print("Feature engineering complete.")
print(f"\nNew columns added: hour, day_of_week, is_weekend, time_period, utilization_rate, revenue_proxy, kwh_per_pile, queue_proxy")

# %%
# quick sanity check on utilization rates
print("\n=== Utilization Rate Statistics ===")
print(urban_long['utilization_rate'].describe())
print(f"\nTimesteps with utilization > 80%: {(urban_long['utilization_rate'] > 0.8).sum():,}")
print(f"Timesteps with utilization < 30%: {(urban_long['utilization_rate'] < 0.3).sum():,}")

# %% [markdown]
# ### 2.5 Rolling Averages (Lagged Features for ML)
# These will be critical for the demand prediction model -- they capture recent trends.
# We compute rolling stats per zone over past 1hr, 3hr, and 6hr windows.

# %%
# sort by zone and time for rolling calculations
urban_long = urban_long.sort_values(['zone_id', 'timestamp']).reset_index(drop=True)

# rolling window sizes in 5-min intervals
WINDOW_1H = 12    # 12 * 5min = 1 hour
WINDOW_3H = 36    # 36 * 5min = 3 hours
WINDOW_6H = 72    # 72 * 5min = 6 hours

# compute rolling features per zone
print("Computing rolling features (this may take a minute)...")

rolling_features = []
for zone in urban_long['zone_id'].unique():
    zone_mask = urban_long['zone_id'] == zone
    zone_data = urban_long.loc[zone_mask, ['utilization_rate', 'volume_kwh']].copy()
    
    # rolling mean utilization
    zone_data['util_roll_1h'] = zone_data['utilization_rate'].rolling(WINDOW_1H, min_periods=1).mean()
    zone_data['util_roll_3h'] = zone_data['utilization_rate'].rolling(WINDOW_3H, min_periods=1).mean()
    zone_data['util_roll_6h'] = zone_data['utilization_rate'].rolling(WINDOW_6H, min_periods=1).mean()
    
    # rolling mean volume
    zone_data['vol_roll_1h'] = zone_data['volume_kwh'].rolling(WINDOW_1H, min_periods=1).mean()
    
    # lagged utilization (for the prediction model)
    zone_data['util_lag_1'] = zone_data['utilization_rate'].shift(1)
    zone_data['util_lag_12'] = zone_data['utilization_rate'].shift(12)   # 1 hour ago
    zone_data['util_lag_288'] = zone_data['utilization_rate'].shift(288) # 1 day ago (288 * 5min = 24hrs)
    
    rolling_features.append(zone_data[['util_roll_1h', 'util_roll_3h', 'util_roll_6h',
                                        'vol_roll_1h', 'util_lag_1', 'util_lag_12', 'util_lag_288']])

rolling_df = pd.concat(rolling_features, axis=0)
urban_long = pd.concat([urban_long, rolling_df.reset_index(drop=True)], axis=1)

print(f"Rolling features added. Shape: {urban_long.shape}")

# %% [markdown]
# ### 2.6 Handle Missing Values

# %%
# check for missing values
print("=== Missing Values ===")
missing = urban_long.isnull().sum()
missing_pct = (missing / len(urban_long) * 100).round(2)
missing_report = pd.DataFrame({'count': missing, 'pct': missing_pct})
print(missing_report[missing_report['count'] > 0])

# %%
# strategy: forward-fill lagged features (NaN at the start of each zone's series)
# these are naturally missing because we can't look back before the data starts
lag_cols = ['util_lag_1', 'util_lag_12', 'util_lag_288', 'util_roll_1h',
            'util_roll_3h', 'util_roll_6h', 'vol_roll_1h']

for col in lag_cols:
    urban_long[col] = urban_long.groupby('zone_id')[col].transform(
        lambda x: x.fillna(method='bfill')
    )

# any remaining nulls -> fill with column median
urban_long[lag_cols] = urban_long[lag_cols].fillna(urban_long[lag_cols].median())

print("Missing values after filling:")
print(urban_long[lag_cols].isnull().sum())

# %% [markdown]
# ### 2.7 Save Processed UrbanEV Data

# %%
# save the full long-format dataframe
urban_long.to_csv(os.path.join(PROCESSED_DIR, 'urbanev_processed.csv'), index=False)
print(f"Saved urbanev_processed.csv -- {urban_long.shape[0]:,} rows, {urban_long.shape[1]} columns")

# also save a smaller hourly-aggregated version for faster EDA
hourly_agg = urban_long.groupby(['zone_id', urban_long['datetime'].dt.floor('h')]).agg({
    'occupancy': 'mean',
    'volume_kwh': 'sum',
    'duration_hrs': 'sum',
    'price_cny_kwh': 'mean',
    'utilization_rate': 'mean',
    'revenue_proxy': 'sum',
    'total_piles': 'first',
    'is_weekend': 'first',
    'time_period': 'first',
    'CBD': 'first',
    'dynamic_pricing': 'first',
    'queue_proxy': 'mean'
}).reset_index()
hourly_agg.columns = ['zone_id', 'datetime'] + list(hourly_agg.columns[2:])
hourly_agg['hour'] = hourly_agg['datetime'].dt.hour
hourly_agg['day_of_week'] = hourly_agg['datetime'].dt.dayofweek

hourly_agg.to_csv(os.path.join(PROCESSED_DIR, 'urbanev_hourly.csv'), index=False)
print(f"Saved urbanev_hourly.csv -- {hourly_agg.shape[0]:,} rows")

# %% [markdown]
# ---
# ## 3. Load and Process ACN Dataset
# The ACN dataset has **16,304 individual charging sessions** from Caltech/JPL with
# connection/disconnection times, energy delivered, and station IDs.

# %%
acn_raw = pd.read_excel(os.path.join(ACN_DIR, 'acndata_sessions.json.xlsx'))
print(f"ACN raw shape: {acn_raw.shape}")
print(f"\nColumns: {list(acn_raw.columns)}")

# %%
# keep only relevant columns and drop metadata rows
acn_cols = ['connectionTime', 'disconnectTime', 'doneChargingTime', 'kWhDelivered',
            'sessionID', 'stationID', 'clusterID', 'spaceID', 'siteID']

acn = acn_raw[acn_cols].copy()

# drop rows where key fields are missing
acn = acn.dropna(subset=['connectionTime', 'disconnectTime', 'kWhDelivered'])
print(f"ACN after dropping missing key fields: {acn.shape}")

# %% [markdown]
# ### 3.1 Parse Timestamps
# ACN timestamps are in format like: `"Wed, 25 Apr 2018 11:08:04 GMT"`

# %%
# parse the date strings
acn['connection_dt'] = pd.to_datetime(acn['connectionTime'], format='%a, %d %b %Y %H:%M:%S GMT', utc=True)
acn['disconnect_dt'] = pd.to_datetime(acn['disconnectTime'], format='%a, %d %b %Y %H:%M:%S GMT', utc=True)

# convert to local time (America/Los_Angeles for Caltech)
acn['connection_dt'] = acn['connection_dt'].dt.tz_convert('America/Los_Angeles')
acn['disconnect_dt'] = acn['disconnect_dt'].dt.tz_convert('America/Los_Angeles')

# also parse doneChargingTime where available
acn['done_charging_dt'] = pd.to_datetime(acn['doneChargingTime'], format='%a, %d %b %Y %H:%M:%S GMT',
                                          utc=True, errors='coerce')
acn.loc[acn['done_charging_dt'].notna(), 'done_charging_dt'] = \
    acn.loc[acn['done_charging_dt'].notna(), 'done_charging_dt'].dt.tz_convert('America/Los_Angeles')

print(f"Date range: {acn['connection_dt'].min()} to {acn['connection_dt'].max()}")
print(f"Total sessions: {len(acn)}")

# %% [markdown]
# ### 3.2 Feature Engineering -- ACN

# %%
# --- session duration ---
acn['session_duration_hrs'] = (acn['disconnect_dt'] - acn['connection_dt']).dt.total_seconds() / 3600

# charging duration (time actually spent charging)
acn['charging_duration_hrs'] = np.where(
    acn['done_charging_dt'].notna(),
    (acn['done_charging_dt'] - acn['connection_dt']).dt.total_seconds() / 3600,
    acn['session_duration_hrs']  # fallback to total session
)

# idle time = plugged in but not charging
acn['idle_time_hrs'] = acn['session_duration_hrs'] - acn['charging_duration_hrs']
acn['idle_time_hrs'] = acn['idle_time_hrs'].clip(lower=0)

# %%
# --- temporal features ---
acn['hour'] = acn['connection_dt'].dt.hour
acn['day_of_week'] = acn['connection_dt'].dt.dayofweek
acn['is_weekend'] = (acn['day_of_week'] >= 5).astype(int)
acn['date'] = acn['connection_dt'].dt.date
acn['month'] = acn['connection_dt'].dt.month

# time period
acn['time_period'] = acn['hour'].apply(classify_period)

# %%
# --- energy features ---
# charging rate (avg kW during the charging window)
acn['avg_charging_rate_kw'] = np.where(
    acn['charging_duration_hrs'] > 0,
    acn['kWhDelivered'] / acn['charging_duration_hrs'],
    0
)

# baseline revenue at flat rate
# The brief mentions an Indian flat-rate baseline. Because ACN is a US workplace
# dataset, revenue is evaluated in USD using a comparable $0.30/kWh baseline.
BASELINE_RATE = 0.30  # $/kWh flat rate for US data
acn['baseline_revenue'] = acn['kWhDelivered'] * BASELINE_RATE

# %%
# --- filter outliers ---
# remove sessions that are clearly erroneous
print(f"Before outlier removal: {len(acn)} sessions")

# filter: session duration must be between 5 min and 48 hours
acn = acn[(acn['session_duration_hrs'] > 5/60) & (acn['session_duration_hrs'] < 48)]

# filter: energy must be positive and reasonable (< 200 kWh)
acn = acn[(acn['kWhDelivered'] > 0) & (acn['kWhDelivered'] < 200)]

# filter: charging rate should be reasonable (< 150 kW for level 2/DC fast)
acn = acn[acn['avg_charging_rate_kw'] < 150]

print(f"After outlier removal: {len(acn)} sessions")

# %%
# quick summary
print("\n=== ACN Session Statistics ===")
print(f"Energy delivered (kWh): mean={acn['kWhDelivered'].mean():.1f}, median={acn['kWhDelivered'].median():.1f}")
print(f"Session duration (hrs): mean={acn['session_duration_hrs'].mean():.1f}, median={acn['session_duration_hrs'].median():.1f}")
print(f"Charging duration (hrs): mean={acn['charging_duration_hrs'].mean():.1f}, median={acn['charging_duration_hrs'].median():.1f}")
print(f"Idle time (hrs): mean={acn['idle_time_hrs'].mean():.1f}")
print(f"Avg charging rate (kW): mean={acn['avg_charging_rate_kw'].mean():.1f}")

# %% [markdown]
# ### 3.3 Build Hourly Demand Profile from ACN Sessions
# We need to know how many sessions are active at each hour -- this tells us station utilization.

# %%
# create hourly demand profile: count of active sessions per hour
date_range = pd.date_range(
    start=acn['connection_dt'].min().floor('h'),
    end=acn['disconnect_dt'].max().ceil('h'),
    freq='h'
)

hourly_demand = []
for hr in date_range:
    hr_end = hr + pd.Timedelta(hours=1)
    # count sessions active during this hour
    active = ((acn['connection_dt'] < hr_end) & (acn['disconnect_dt'] > hr)).sum()
    # sum energy being delivered during this hour (approximate)
    active_sessions = acn[(acn['connection_dt'] < hr_end) & (acn['disconnect_dt'] > hr)]
    kwh_approx = active_sessions['avg_charging_rate_kw'].sum()  # total kW being drawn
    
    hourly_demand.append({
        'datetime': hr,
        'active_sessions': active,
        'total_kw_demand': kwh_approx
    })

acn_hourly = pd.DataFrame(hourly_demand)
acn_hourly['hour'] = acn_hourly['datetime'].dt.hour
acn_hourly['day_of_week'] = acn_hourly['datetime'].dt.dayofweek
acn_hourly['is_weekend'] = (acn_hourly['day_of_week'] >= 5).astype(int)
acn_hourly['time_period'] = acn_hourly['hour'].apply(classify_period)
acn_hourly['date'] = acn_hourly['datetime'].dt.date

print(f"ACN hourly demand profile: {acn_hourly.shape}")
print(acn_hourly.head())

# %% [markdown]
# ### 3.4 Save Processed ACN Data

# %%
# save session-level data
acn_save_cols = ['sessionID', 'stationID', 'clusterID', 'connection_dt', 'disconnect_dt',
                 'done_charging_dt', 'kWhDelivered', 'session_duration_hrs',
                 'charging_duration_hrs', 'idle_time_hrs', 'avg_charging_rate_kw',
                 'baseline_revenue', 'hour', 'day_of_week', 'is_weekend', 'date',
                 'month', 'time_period']

acn[acn_save_cols].to_csv(os.path.join(PROCESSED_DIR, 'acn_sessions_processed.csv'), index=False)
print(f"Saved acn_sessions_processed.csv -- {len(acn)} sessions")

# save hourly demand profile
acn_hourly.to_csv(os.path.join(PROCESSED_DIR, 'acn_hourly_demand.csv'), index=False)
print(f"Saved acn_hourly_demand.csv -- {len(acn_hourly)} hourly records")

# %% [markdown]
# ---
# ## 4. Data Quality Summary

# %%
print("=" * 60)
print("DATA PREPROCESSING SUMMARY")
print("=" * 60)
print(f"\n--- UrbanEV (Shenzhen) ---")
print(f"  Zones: {urban_long['zone_id'].nunique()}")
print(f"  Time span: {urban_long['datetime'].min()} to {urban_long['datetime'].max()}")
print(f"  Total records: {len(urban_long):,}")
print(f"  Features: {urban_long.shape[1]}")
print(f"  Zones with dynamic pricing: {info_df['dynamic_pricing'].sum()}")
print(f"  Total charging piles: {info_df['count'].sum()}")

print(f"\n--- ACN (Caltech/JPL) ---")
print(f"  Sessions: {len(acn):,}")
print(f"  Unique stations: {acn['stationID'].nunique()}")
print(f"  Date range: {acn['date'].min()} to {acn['date'].max()}")
print(f"  Avg energy/session: {acn['kWhDelivered'].mean():.1f} kWh")

print(f"\n--- Processed Files Saved ---")
for f in os.listdir(PROCESSED_DIR):
    fpath = os.path.join(PROCESSED_DIR, f)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  {f}: {size_mb:.1f} MB")

print("\n[OK] Preprocessing complete. Ready for EDA.")
