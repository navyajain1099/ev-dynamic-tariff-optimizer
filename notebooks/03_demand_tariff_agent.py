# %% [markdown]
# # Notebook 3: Demand Prediction + Tariff Pricing Agent
# **EV Dynamic Tariff Optimization -- Agentic AI Framework**
#
# This notebook implements two merged agents:
# - **Agent 1 (Demand Predictor):** ML model trained on UrbanEV to forecast utilization
# - **Agent 2 (Tariff Pricing):** Dynamic pricing logic applied to both datasets
#
# The agents are merged because the pricing decision directly depends on the demand forecast --
# keeping them together creates a cleaner, more efficient pipeline.

# %% [markdown]
# ## 1. Setup

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so plots save without blocking
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, using RandomForest as primary model")

try:
    from lightgbm import LGBMRegressor, log_evaluation as lgb_log_callback
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, skipping comparison")

plt.rcParams.update({
    'figure.figsize': (12, 6), 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'figure.dpi': 100
})

COLORS = {'peak': '#e74c3c', 'shoulder': '#f39c12', 'off_peak': '#2ecc71',
          'primary': '#3498db', 'secondary': '#9b59b6', 'dark': '#2c3e50'}

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
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# load processed data
urban = pd.read_csv(os.path.join(PROCESSED_DIR, 'urbanev_processed.csv'), parse_dates=['datetime'])
acn = pd.read_csv(os.path.join(PROCESSED_DIR, 'acn_sessions_processed.csv'), parse_dates=['connection_dt', 'disconnect_dt'])

print(f"UrbanEV: {urban.shape[0]:,} records")
print(f"ACN: {acn.shape[0]:,} sessions")

# %% [markdown]
# ---
# ## 2. Agent 1: Demand Prediction Model
#
# **Objective:** Predict charger utilization rate 1 hour ahead for each zone.
#
# **Prediction horizon:** 1 hour (12 steps of 5-minute intervals)
#
# **Target variable:** `utilization_rate` shifted 12 steps forward (1-hour ahead forecast)
#
# **Features:** temporal (hour, day_of_week, is_weekend), lagged utilization, rolling means, zone info
#
# **Note on util_lag_1:** In this framing, `util_lag_1` represents the *current* observed
# utilization at prediction time (t=0), and we are forecasting t+12 (one hour ahead).
# In a real deployment, this would be the live sensor reading from the station.
# This is a standard approach in operational forecasting (e.g., electricity load forecasting).

# %% [markdown]
# ### 2.1 Feature Selection and Preparation

# %%
# sort by zone and time to ensure correct lag alignment
urban = urban.sort_values(['zone_id', 'datetime']).reset_index(drop=True)

# ---------------------------------------------------------------------
# FIX: Create a 1-hour-ahead target instead of predicting the same step
# util_lag_1 = current utilization (t=0), target = utilization at t+12
# This makes the problem a genuine 1-hour-ahead forecast, not a trivial
# persistence model. util_lag_1 is now a valid input feature (current state).
# ---------------------------------------------------------------------
FORECAST_HORIZON = 12  # 12 x 5-min intervals = 1 hour ahead

urban['target_utilization'] = urban.groupby('zone_id')['utilization_rate'].shift(-FORECAST_HORIZON)

feature_cols = [
    'hour', 'day_of_week', 'is_weekend',
    'total_piles', 'fast_piles', 'slow_piles',
    'CBD', 'dynamic_pricing',
    'price_cny_kwh',
    'util_lag_1',    # current observed utilization (valid input for 1-hr ahead forecast)
    'util_lag_12',   # utilization 1 hour ago
    'util_lag_288',  # utilization 24 hours ago (same time yesterday)
    'util_roll_1h',  # rolling mean last 1 hour
    'util_roll_3h',  # rolling mean last 3 hours
    'util_roll_6h',  # rolling mean last 6 hours
    'vol_roll_1h',   # rolling energy volume last 1 hour
    'volume_kwh',    # energy in current interval
    'duration_hrs'   # session duration proxy
]

target_col = 'target_utilization'  # 1-hour-ahead utilization (was 'utilization_rate' before)

# drop rows with missing values (NaNs created by shift at end of each zone's series)
model_data = urban[feature_cols + [target_col, 'datetime', 'zone_id',
                                    'time_period', 'utilization_rate']].dropna()
print(f"Model data shape: {model_data.shape}")
print(f"\nForecast horizon: {FORECAST_HORIZON} steps = 1 hour ahead")
print(f"\nFeature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# %% [markdown]
# ### 2.2 Train/Test Split (Chronological)
# We use the last 6 days as the test set to prevent data leakage -- the model must
# predict *future* demand, so we cannot randomly split.

# %%
# chronological split: first ~80% for training, last ~20% for testing
split_date = model_data['datetime'].max() - pd.Timedelta(days=6)
train_mask = model_data['datetime'] < split_date
test_mask  = model_data['datetime'] >= split_date

X_train = model_data.loc[train_mask, feature_cols]
y_train = model_data.loc[train_mask, target_col]
X_test  = model_data.loc[test_mask,  feature_cols]
y_test  = model_data.loc[test_mask,  target_col]

print(f"Training set: {len(X_train):,} samples (up to {split_date.date()})")
print(f"Test set:     {len(X_test):,}  samples (from {split_date.date()} onwards)")
print(f"Train/test ratio: {len(X_train)/len(model_data)*100:.1f}% / {len(X_test)/len(model_data)*100:.1f}%")

# %% [markdown]
# ### 2.3 Model Training and Comparison

# %%
models  = {}
results = {}

# --- Model 1: Random Forest ---
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=10,
                           random_state=42, n_jobs=-1, verbose=2)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
models['Random Forest'] = rf
results['Random Forest'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
    'MAE':  mean_absolute_error(y_test, rf_pred),
    'R2':   r2_score(y_test, rf_pred),
    'predictions': rf_pred
}
print(f"  RMSE: {results['Random Forest']['RMSE']:.4f}")
print(f"  MAE:  {results['Random Forest']['MAE']:.4f}")
print(f"  R2:   {results['Random Forest']['R2']:.4f}")

# %%
# --- Model 2: XGBoost ---
if HAS_XGBOOST:
    print("\nTraining XGBoost...")
    xgb = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       min_child_weight=5, random_state=42, n_jobs=-1,
                       verbosity=0)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)
    xgb_pred = xgb.predict(X_test)
    models['XGBoost'] = xgb
    results['XGBoost'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'MAE':  mean_absolute_error(y_test, xgb_pred),
        'R2':   r2_score(y_test, xgb_pred),
        'predictions': xgb_pred
    }
    print(f"  RMSE: {results['XGBoost']['RMSE']:.4f}")
    print(f"  MAE:  {results['XGBoost']['MAE']:.4f}")
    print(f"  R2:   {results['XGBoost']['R2']:.4f}")

# %%
# --- Model 3: LightGBM ---
if HAS_LGBM:
    print("\nTraining LightGBM...")
    lgbm = LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=5, random_state=42, n_jobs=-1,
                          verbose=-1)
    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             callbacks=[lgb_log_callback(10)])
    lgbm_pred = lgbm.predict(X_test)
    models['LightGBM'] = lgbm
    results['LightGBM'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, lgbm_pred)),
        'MAE':  mean_absolute_error(y_test, lgbm_pred),
        'R2':   r2_score(y_test, lgbm_pred),
        'predictions': lgbm_pred
    }
    print(f"  RMSE: {results['LightGBM']['RMSE']:.4f}")
    print(f"  MAE:  {results['LightGBM']['MAE']:.4f}")
    print(f"  R2:   {results['LightGBM']['R2']:.4f}")

# %% [markdown]
# ### 2.4 Model Comparison

# %%
comparison_df = pd.DataFrame(results).T[['RMSE', 'MAE', 'R2']]
comparison_df = comparison_df.sort_values('R2', ascending=False)
print("\n=== MODEL COMPARISON ===")
print(comparison_df.to_string())

best_model_name = comparison_df['R2'].idxmax()
best_model      = models[best_model_name]
best_preds      = results[best_model_name]['predictions']
print(f"\nBest model: {best_model_name} (R2 = {comparison_df.loc[best_model_name, 'R2']:.4f})")

comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'))

# %%
# --- Visualization: actual vs predicted ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(y_test.values[:5000], best_preds[:5000], alpha=0.1, s=5, color=COLORS['primary'])
axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Utilization (t+1hr)')
axes[0].set_ylabel('Predicted Utilization (t+1hr)')
axes[0].set_title(f'{best_model_name}: Actual vs Predicted (1-hr ahead)')
axes[0].legend()

residuals = y_test.values - best_preds
axes[1].hist(residuals, bins=50, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
axes[1].axvline(x=0, color='red', linestyle='--')
axes[1].set_xlabel('Residual (Actual - Predicted)')
axes[1].set_ylabel('Count')
axes[1].set_title('Residual Distribution')

sample_zone = model_data.loc[test_mask, 'zone_id'].value_counts().index[0]
zone_mask   = (model_data['zone_id'] == sample_zone) & test_mask
zone_actual = model_data.loc[zone_mask, target_col].values[:288]
zone_pred   = best_model.predict(model_data.loc[zone_mask, feature_cols].head(288))
axes[2].plot(range(len(zone_actual)), zone_actual, label='Actual (t+1hr)',
             color=COLORS['dark'], linewidth=1.5)
axes[2].plot(range(len(zone_pred)),   zone_pred,   label='Predicted (t+1hr)',
             color=COLORS['peak'], linewidth=1.5, linestyle='--')
axes[2].set_xlabel('Time Steps (5-min intervals)')
axes[2].set_ylabel('Utilization Rate')
axes[2].set_title(f'Zone {sample_zone}: 1-Hour-Ahead Forecast')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'demand_prediction_results.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 2.5 Feature Importance
#
# **Interpretation note:** `util_lag_1` (current utilization) is expected to be the most
# important feature -- this is consistent with time-series forecasting literature where
# the current state is the strongest predictor of the near-future state. The model still
# extracts meaningful signal from temporal features (hour, day_of_week) and rolling
# averages, which drive the *deviation* from the current state.

# %%
importances = best_model.feature_importances_

importance_df = pd.DataFrame({
    'feature':    feature_cols,
    'importance': importances
}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(importance_df['feature'], importance_df['importance'], color=COLORS['primary'])
ax.set_xlabel('Feature Importance')
ax.set_title(f'{best_model_name} -- Feature Importance\n(util_lag_1 = current observed state at prediction time)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 5 most important features:")
for _, row in importance_df.tail(5).iterrows():
    print(f"  {row['feature']:>20s}: {row['importance']:.4f}")

# %% [markdown]
# ### 2.6 Save Demand Predictions

# %%
pred_df = model_data.loc[test_mask, ['datetime', 'zone_id',
                                      'utilization_rate', target_col]].copy()
pred_df.rename(columns={target_col: 'actual_future_utilization'}, inplace=True)
pred_df['predicted_utilization'] = best_preds
pred_df['prediction_error']      = pred_df['actual_future_utilization'] - pred_df['predicted_utilization']
pred_df.to_csv(os.path.join(OUTPUT_DIR, 'demand_predictions.csv'), index=False)
print(f"Saved {len(pred_df):,} predictions to demand_predictions.csv")
print("  Column 'predicted_utilization' = 1-hour-ahead forecast")
print("  Column 'utilization_rate'      = current utilization at prediction time")

# %% [markdown]
# ---
# ## 3. Agent 2: Tariff Pricing Agent
#
# **Objective:** Use 1-hour-ahead demand predictions to set dynamic prices proactively.
#
# **Design choices:**
# - Base rate: CNY 1.0/kWh for UrbanEV (Shenzhen, China), USD 0.30/kWh for ACN (USA)
#   These are kept in their native currencies and NOT mixed.
# - Surge:    utilization > 75% → gradually increase price up to +40%
# - Discount: utilization < 30% → gradually decrease price down to -30%
# - Normal:   30-75% → base rate with minor time-of-day nudge
# - Price cap: ±40% max (no 2x spikes, per organizer guidance)
#
# **Why proactive pricing works:** Because we predict 1 hour ahead, the Tariff Agent
# can set prices *before* congestion occurs, giving users time to respond.

# %% [markdown]
# ### 3.1 Define Pricing Function

# %%
def calculate_dynamic_price(utilization, base_rate, time_period='shoulder'):
    """
    Calculate dynamic price based on predicted 1-hour-ahead utilization.

    Graduated pricing:
    - High demand (>75%): surge, scales linearly up to 1.40x base
    - Low demand (<30%):  discount, scales linearly down to 0.70x base
    - Normal (30-75%):    base rate with minor time-of-day adjustment

    Parameters
    ----------
    utilization  : predicted utilization rate (0-1), 1 hour ahead
    base_rate    : base price per kWh (CNY for UrbanEV, USD for ACN)
    time_period  : 'peak', 'shoulder', or 'off_peak'

    Returns
    -------
    dynamic_price : adjusted price per kWh (same currency as base_rate)
    multiplier    : price multiplier applied
    pricing_tier  : 'surge', 'normal', or 'discount'
    """
    SURGE_THRESHOLD    = 0.75
    DISCOUNT_THRESHOLD = 0.30
    MAX_SURGE          = 1.40   # max +40% above base
    MAX_DISCOUNT       = 0.70   # max -30% below base

    if utilization > SURGE_THRESHOLD:
        surge_intensity = min((utilization - SURGE_THRESHOLD) / (1.0 - SURGE_THRESHOLD), 1.0)
        multiplier = 1.0 + (MAX_SURGE - 1.0) * surge_intensity
        tier = 'surge'

    elif utilization < DISCOUNT_THRESHOLD:
        discount_intensity = min((DISCOUNT_THRESHOLD - utilization) / DISCOUNT_THRESHOLD, 1.0)
        multiplier = 1.0 - (1.0 - MAX_DISCOUNT) * discount_intensity
        tier = 'discount'

    else:
        # small nudge within normal range based on time of day
        if time_period == 'peak':
            multiplier = 1.05
        elif time_period == 'off_peak':
            multiplier = 0.95
        else:
            multiplier = 1.00
        tier = 'normal'

    return base_rate * multiplier, multiplier, tier

# quick demo
print("=== Pricing Function Demo (CNY 1.0/kWh base, UrbanEV) ===")
for util in [0.10, 0.25, 0.40, 0.60, 0.75, 0.85, 0.95]:
    price, mult, tier = calculate_dynamic_price(util, 1.0)
    print(f"  Utilization {util:.0%} -> {tier:>8s} | multiplier: {mult:.2f}x | price: {price:.2f} CNY/kWh")

# %% [markdown]
# ### 3.2 Apply Dynamic Pricing to UrbanEV Data
# Uses the **predicted_utilization** (1-hour-ahead forecast) as input to the pricing agent.

# %%
print("Applying dynamic pricing to UrbanEV predictions (using 1-hr ahead forecast)...")

urbanev_pricing = pred_df.copy()
urbanev_pricing = urbanev_pricing.merge(
    model_data.loc[test_mask, ['datetime', 'zone_id', 'price_cny_kwh', 'volume_kwh',
                                'total_piles', 'time_period']],
    on=['datetime', 'zone_id'], how='left'
)

# apply pricing using PREDICTED (future) utilization -- this is the key agentic decision
pricing_results = urbanev_pricing.apply(
    lambda row: calculate_dynamic_price(
        row['predicted_utilization'], base_rate=1.0,
        time_period=row.get('time_period', 'shoulder')
    ), axis=1
)
urbanev_pricing['dynamic_price']    = [r[0] for r in pricing_results]
urbanev_pricing['price_multiplier'] = [r[1] for r in pricing_results]
urbanev_pricing['pricing_tier']     = [r[2] for r in pricing_results]

# revenues
urbanev_pricing['baseline_revenue'] = urbanev_pricing['volume_kwh'] * 1.0
urbanev_pricing['dynamic_revenue']  = urbanev_pricing['volume_kwh'] * urbanev_pricing['dynamic_price']

print(f"\nPricing tier distribution (UrbanEV):")
print(urbanev_pricing['pricing_tier'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))

# %% [markdown]
# ### 3.3 Apply Dynamic Pricing to ACN Sessions
# ACN is used for **revenue metrics** (USD). Utilization is estimated from hourly demand profiles.

# %%
acn_hourly = pd.read_csv(os.path.join(PROCESSED_DIR, 'acn_hourly_demand.csv'),
                          parse_dates=['datetime'])

# utilization proxy: active sessions / peak capacity
max_capacity = acn_hourly['active_sessions'].quantile(0.95)
print(f"ACN estimated max capacity (95th pct): {max_capacity:.0f} simultaneous sessions")

if 'utilization_proxy' not in acn_hourly.columns:
    acn_hourly['utilization_proxy'] = (acn_hourly['active_sessions'] / max_capacity).clip(0, 1)

acn_util_map = acn_hourly.groupby('hour')['utilization_proxy'].mean().to_dict()
acn['estimated_utilization'] = acn['hour'].map(acn_util_map).fillna(0.3)

# -----------------------------------------------------------------------
# ACN base rate: USD 0.30/kWh (US workplace charging market rate)
# NOTE: All ACN revenue figures are in USD. UrbanEV figures are in CNY.
#       Currencies are NOT mixed anywhere in this analysis.
# -----------------------------------------------------------------------
BASELINE_RATE_USD = 0.30

acn['baseline_revenue'] = acn['kWhDelivered'] * BASELINE_RATE_USD

# ACN-specific pricing function
# ACN is a workplace charging dataset with naturally low utilization,
# so we calibrate the discount threshold lower (15%) to avoid
# unnecessarily discounting sessions that are simply normal workplace behavior.
def calculate_dynamic_price_acn(utilization, base_rate, time_period='shoulder'):
    SURGE_THRESHOLD    = 0.75
    DISCOUNT_THRESHOLD = 0.15   # lower than UrbanEV (0.30) -- ACN baseline util is naturally low
    MAX_SURGE          = 1.40
    MAX_DISCOUNT       = 0.85   # smaller max discount on ACN (only -15%)

    if utilization > SURGE_THRESHOLD:
        surge_intensity = min((utilization - SURGE_THRESHOLD) / (1.0 - SURGE_THRESHOLD), 1.0)
        multiplier = 1.0 + (MAX_SURGE - 1.0) * surge_intensity
        tier = 'surge'
    elif utilization < DISCOUNT_THRESHOLD:
        discount_intensity = min((DISCOUNT_THRESHOLD - utilization) / DISCOUNT_THRESHOLD, 1.0)
        multiplier = 1.0 - (1.0 - MAX_DISCOUNT) * discount_intensity
        tier = 'discount'
    else:
        if time_period == 'peak':
            multiplier = 1.10   # stronger peak premium for ACN
        elif time_period == 'off_peak':
            multiplier = 0.97
        else:
            multiplier = 1.00
        tier = 'normal'

    return base_rate * multiplier, multiplier, tier

pricing_results_acn = acn.apply(
    lambda row: calculate_dynamic_price_acn(
        row['estimated_utilization'], BASELINE_RATE_USD, row['time_period']
    ), axis=1
)
acn['dynamic_price']    = [r[0] for r in pricing_results_acn]
acn['price_multiplier'] = [r[1] for r in pricing_results_acn]
acn['pricing_tier']     = [r[2] for r in pricing_results_acn]
acn['dynamic_revenue']  = acn['kWhDelivered'] * acn['dynamic_price']

print(f"\nACN Pricing tier distribution:")
print(acn['pricing_tier'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))

# %% [markdown]
# ### 3.4 Revenue Impact Analysis (ACN -- USD)

# %%
baseline_total = acn['baseline_revenue'].sum()
dynamic_total  = acn['dynamic_revenue'].sum()
revenue_gain_pct = ((dynamic_total - baseline_total) / baseline_total) * 100

print("=" * 60)
print("REVENUE ANALYSIS -- ACN Dataset (USD)")
print("=" * 60)
print(f"  Baseline Revenue (flat $0.30/kWh): ${baseline_total:,.2f}")
print(f"  Dynamic Revenue:                   ${dynamic_total:,.2f}")
print(f"  Revenue Gain:                      ${dynamic_total - baseline_total:,.2f}")
print(f"  Revenue Gain %:                    {revenue_gain_pct:+.2f}%")

print("\n  Revenue by Time Period (USD):")
for period in ['peak', 'shoulder', 'off_peak']:
    mask = acn['time_period'] == period
    base = acn.loc[mask, 'baseline_revenue'].sum()
    dyn  = acn.loc[mask, 'dynamic_revenue'].sum()
    gain = ((dyn - base) / base * 100) if base > 0 else 0
    print(f"    {period:>10s}: ${base:>10,.2f} -> ${dyn:>10,.2f} ({gain:+.1f}%)")

# %% [markdown]
# ### 3.5 Utilization & Off-Peak Analysis (UrbanEV)
#
# **Demand response simulation:** We model how customers react to price changes using
# price elasticity of demand = -0.30. This is a conservative, well-cited value from
# EV charging literature. A 10% price increase leads to a ~3% demand decrease.
#
# Formula: demand_change = elasticity × (price_multiplier - 1)
# Simulated_utilization = actual_utilization × (1 - demand_change)
#
# This replaces the arbitrary 0.15 coefficient from the original version.

# %%
PRICE_ELASTICITY = -0.30  # conservative EV demand elasticity

print("=" * 60)
print("UTILIZATION ANALYSIS -- UrbanEV Dataset (CNY)")
print("=" * 60)

# simulate demand response using elasticity (consistent with monitoring agent)
urbanev_pricing['demand_change_pct'] = PRICE_ELASTICITY * (urbanev_pricing['price_multiplier'] - 1)
urbanev_pricing['simulated_utilization'] = (
    urbanev_pricing['utilization_rate'] * (1 + urbanev_pricing['demand_change_pct'])
).clip(0, 1)

baseline_util = urbanev_pricing['utilization_rate'].mean()
simulated_util_mean = urbanev_pricing['simulated_utilization'].mean()

print(f"  Baseline avg utilization:  {baseline_util:.4f} ({baseline_util*100:.1f}%)")
print(f"  Simulated avg utilization: {simulated_util_mean:.4f} ({simulated_util_mean*100:.1f}%)")
print(f"  Utilization change:        {(simulated_util_mean - baseline_util)*100:+.2f} pp")

# --- Off-Peak Uplift ---
# discount sessions = predicted utilization < 30% → price cut → demand should rise
off_peak_mask = urbanev_pricing['pricing_tier'] == 'discount'
off_peak_baseline   = urbanev_pricing.loc[off_peak_mask, 'utilization_rate'].mean()
off_peak_simulated  = urbanev_pricing.loc[off_peak_mask, 'simulated_utilization'].mean()
off_peak_uplift     = ((off_peak_simulated - off_peak_baseline) / off_peak_baseline * 100) if off_peak_baseline > 0 else 0

print(f"\n  Off-Peak Uplift (discount tier sessions):")
print(f"    Baseline off-peak utilization:  {off_peak_baseline:.4f} ({off_peak_baseline*100:.1f}%)")
print(f"    Simulated off-peak utilization: {off_peak_simulated:.4f} ({off_peak_simulated*100:.1f}%)")
print(f"    Off-Peak Uplift:                {off_peak_uplift:+.1f}%")
print(f"    (Elasticity-based: discount reduces price → demand increases)")

# %% [markdown]
# ### 3.6 Pricing Visualization

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
tier_colors = {'surge': COLORS['peak'], 'normal': COLORS['shoulder'], 'discount': COLORS['off_peak']}

# 1. price multiplier distribution
for tier in ['surge', 'normal', 'discount']:
    mask = urbanev_pricing['pricing_tier'] == tier
    axes[0, 0].hist(urbanev_pricing.loc[mask, 'price_multiplier'], bins=30, alpha=0.7,
                     color=tier_colors[tier], label=f'{tier.title()} ({mask.sum():,})')
axes[0, 0].set_xlabel('Price Multiplier')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Dynamic Price Multiplier Distribution (UrbanEV)')
axes[0, 0].legend()

# 2. hourly average price multiplier
hourly_mult = urbanev_pricing.groupby(urbanev_pricing['datetime'].dt.hour)['price_multiplier'].mean()
bar_colors = []
for h in hourly_mult.index:
    if h in range(8, 12) or h in range(17, 21):
        bar_colors.append(COLORS['peak'])
    elif h in range(23, 24) or h in range(0, 6):
        bar_colors.append(COLORS['off_peak'])
    else:
        bar_colors.append(COLORS['shoulder'])
axes[0, 1].bar(hourly_mult.index, hourly_mult.values, color=bar_colors, edgecolor='white')
axes[0, 1].axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Base rate')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Average Price Multiplier')
axes[0, 1].set_title('Hourly Average Price Multiplier (UrbanEV)')
axes[0, 1].legend()

# 3. revenue comparison by period (ACN -- USD)
periods   = ['peak', 'shoulder', 'off_peak']
base_revs = [acn.loc[acn['time_period'] == p, 'baseline_revenue'].sum() for p in periods]
dyn_revs  = [acn.loc[acn['time_period'] == p, 'dynamic_revenue'].sum()  for p in periods]
x = np.arange(len(periods))
width = 0.35
axes[1, 0].bar(x - width/2, base_revs, width, label='Baseline ($)', color=COLORS['dark'],    alpha=0.7)
axes[1, 0].bar(x + width/2, dyn_revs,  width, label='Dynamic ($)',  color=COLORS['primary'], alpha=0.7)
axes[1, 0].set_xlabel('Time Period')
axes[1, 0].set_ylabel('Total Revenue (USD)')
axes[1, 0].set_title('Revenue: Baseline vs Dynamic (ACN, USD)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(['Peak', 'Shoulder', 'Off-Peak'])
axes[1, 0].legend()

# 4. utilization vs price multiplier scatter
sample = urbanev_pricing.sample(min(5000, len(urbanev_pricing)), random_state=42)
scatter_colors = [tier_colors.get(t, 'gray') for t in sample['pricing_tier']]
axes[1, 1].scatter(sample['predicted_utilization'], sample['price_multiplier'],
                    c=scatter_colors, alpha=0.3, s=10)
axes[1, 1].set_xlabel('Predicted Utilization (1-hr ahead)')
axes[1, 1].set_ylabel('Price Multiplier')
axes[1, 1].set_title('Predicted Utilization vs Price Multiplier (UrbanEV)')
axes[1, 1].axhline(y=1.0,  color='black', linestyle='--', alpha=0.5)
axes[1, 1].axvline(x=0.75, color='red',   linestyle='--', alpha=0.5, label='Surge threshold (75%)')
axes[1, 1].axvline(x=0.30, color='green', linestyle='--', alpha=0.5, label='Discount threshold (30%)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tariff_pricing_results.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 3.7 Save Pricing Outputs

# %%
urbanev_pricing.to_csv(os.path.join(OUTPUT_DIR, 'dynamic_tariffs_urbanev.csv'), index=False)

acn_pricing_save = acn[['sessionID', 'stationID', 'connection_dt', 'kWhDelivered',
                          'hour', 'time_period', 'estimated_utilization',
                          'dynamic_price', 'price_multiplier', 'pricing_tier',
                          'baseline_revenue', 'dynamic_revenue']].copy()
acn_pricing_save.to_csv(os.path.join(OUTPUT_DIR, 'dynamic_tariffs_acn.csv'), index=False)

revenue_summary = pd.DataFrame({
    'Metric': [
        'Baseline Revenue (USD)', 'Dynamic Revenue (USD)', 'Revenue Gain (USD)',
        'Revenue Gain %', 'Surge Sessions % (ACN)', 'Discount Sessions % (ACN)',
        'Off-Peak Uplift % (UrbanEV)', 'Avg Utilization Change (UrbanEV pp)',
        'Price Elasticity Assumption'
    ],
    'Value': [
        f"${baseline_total:,.2f}", f"${dynamic_total:,.2f}",
        f"${dynamic_total - baseline_total:,.2f}", f"{revenue_gain_pct:+.2f}%",
        f"{(acn['pricing_tier'] == 'surge').mean()*100:.1f}%",
        f"{(acn['pricing_tier'] == 'discount').mean()*100:.1f}%",
        f"{off_peak_uplift:+.1f}%",
        f"{(simulated_util_mean - baseline_util)*100:+.2f} pp",
        "-0.30 (conservative, from EV charging literature)"
    ]
})
revenue_summary.to_csv(os.path.join(OUTPUT_DIR, 'revenue_comparison.csv'), index=False)
print("Saved pricing outputs to outputs/")

# %% [markdown]
# ### 3.8 Agent 1+2 Summary

# %%
print("=" * 60)
print("AGENT 1+2 SUMMARY")
print("=" * 60)
print(f"\n--- Demand Prediction (Agent 1) ---")
print(f"  Task:        1-hour-ahead utilization forecasting")
print(f"  Best Model:  {best_model_name}")
print(f"  RMSE: {results[best_model_name]['RMSE']:.4f}")
print(f"  MAE:  {results[best_model_name]['MAE']:.4f}")
print(f"  R2:   {results[best_model_name]['R2']:.4f}")
print(f"\n--- Tariff Pricing (Agent 2) ---")
print(f"  Dataset for Revenue Gain:  ACN (USD)")
print(f"  Dataset for Utilization:   UrbanEV (CNY)")
print(f"  Revenue Gain %:      {revenue_gain_pct:+.2f}%")
print(f"  Off-Peak Uplift:     {off_peak_uplift:+.1f}%")
print(f"  Surge threshold:     75%")
print(f"  Discount threshold:  30%")
print(f"  Max surge:           +40% (1.40x)")
print(f"  Max discount:        -30% (0.70x)")
print(f"  Elasticity assumed:  -0.30")
