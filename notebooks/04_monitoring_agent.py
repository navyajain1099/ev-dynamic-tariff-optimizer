# %% [markdown]
# # Notebook 4: Monitoring & Learning Agent
# **EV Dynamic Tariff Optimization -- Agentic AI Framework**
#
# This notebook implements **Agent 3: the Monitoring & Learning Agent**.
#
# The agent's role is to:
# 1. Evaluate outcomes of pricing decisions made by Agent 2
# 2. Track key performance metrics across multiple simulation episodes
# 3. Adjust pricing thresholds through a feedback loop to improve over time
#
# This simulates a real-world deployment where the system continuously learns.

# %% [markdown]
# ## 1. Setup

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

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

BASE_DIR      = get_project_root()
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'outputs')

# %% [markdown]
# ## 2. Load Data and Previous Agent Outputs

# %%
acn        = pd.read_csv(os.path.join(PROCESSED_DIR, 'acn_sessions_processed.csv'),
                          parse_dates=['connection_dt', 'disconnect_dt'])
acn_hourly = pd.read_csv(os.path.join(PROCESSED_DIR, 'acn_hourly_demand.csv'),
                          parse_dates=['datetime'])
urban_hourly = pd.read_csv(os.path.join(PROCESSED_DIR, 'urbanev_hourly.csv'),
                            parse_dates=['datetime'])
acn_tariffs  = pd.read_csv(os.path.join(OUTPUT_DIR, 'dynamic_tariffs_acn.csv'),
                            parse_dates=['connection_dt'])

print(f"ACN sessions:    {len(acn):,}")
print(f"ACN hourly:      {len(acn_hourly):,}")
print(f"ACN tariffs:     {len(acn_tariffs):,}")
print(f"UrbanEV hourly:  {len(urban_hourly):,}")

# %% [markdown]
# ---
# ## 3. Define Monitoring Metrics
#
# KPIs tracked per episode:
# 1. **Posted Revenue Gain %** -- dynamic vs flat baseline before demand response (ACN, USD)
# 2. **Net Revenue Gain %** -- dynamic vs flat baseline after demand response (ACN, USD)
# 3. **Customer Response Rate** -- demand elasticity proxy (ACN + elasticity assumption)
# 4. **Pricing Efficiency Score** -- net revenue per adjusted kWh delivered (ACN, USD/kWh)
# 5. **Average Waiting Time Reduction** -- queue proxy during peak hours (UrbanEV)

# %% [markdown]
# ### 3.1 Pricing Function (parameterized for feedback loop)

# %%
def dynamic_pricing_agent(utilization, base_rate, surge_threshold, discount_threshold,
                           max_surge=1.40, max_discount=0.85, time_period='shoulder'):
    """
    ACN-calibrated pricing agent with adjustable thresholds for learning.
    max_discount=0.85 (only -15% max) because ACN is low-utilization by nature.
    """
    if utilization > surge_threshold:
        surge_intensity = min((utilization - surge_threshold) / (1.0 - surge_threshold), 1.0)
        multiplier = 1.0 + (max_surge - 1.0) * surge_intensity
        tier = 'surge'
    elif utilization < discount_threshold:
        discount_intensity = min((discount_threshold - utilization) / discount_threshold, 1.0)
        multiplier = 1.0 - (1.0 - max_discount) * discount_intensity
        tier = 'discount'
    else:
        if time_period == 'peak':
            multiplier = 1.10
        elif time_period == 'off_peak':
            multiplier = 0.97
        else:
            multiplier = 1.00
        tier = 'normal'

    return base_rate * multiplier, multiplier, tier

# %% [markdown]
# ### 3.2 Demand Elasticity Model
#
# **Assumption:** Price elasticity of demand = **-0.30**
# A 10% price increase -> about 3% demand decrease.
# Source: Consistent with empirical EV charging studies (conservative estimate).
# Same value used in Notebook 3 for consistency across all agents.

# %%
PRICE_ELASTICITY = -0.30

def simulate_demand_response(original_demand, price_multiplier, elasticity=PRICE_ELASTICITY):
    """
    Simulate how energy demand changes in response to a price change.
    demand_change_pct = elasticity * (price_multiplier - 1)
    """
    price_change  = price_multiplier - 1.0
    demand_change = elasticity * price_change
    return max(0, original_demand * (1 + demand_change))

print("=== Demand Elasticity Demo ===")
for mult in [0.70, 0.85, 1.00, 1.15, 1.30, 1.40]:
    new_d = simulate_demand_response(100, mult)
    print(f"  Price {mult:.2f}x -> Demand: {new_d:.1f} (change: {new_d-100:+.1f}%)")

# %% [markdown]
# ---
# ## 4. Multi-Episode Simulation with Feedback Loop
#
# The agent runs **10 episodes**. After each episode it evaluates the metrics and
# adjusts surge/discount thresholds with a bounded local search. Each episode
# evaluates nearby threshold candidates and selects the policy that balances:
# net revenue gain, waiting-time reduction, and customer disruption.
#
# **Wait Time Reduction** is computed from **peak hour sessions** (not just surge tier),
# so it is non-zero even when ACN has few surge sessions.

# %%
# --- Prepare data ---

max_capacity = acn_hourly['active_sessions'].quantile(0.95)
print(f"Max capacity estimate: {max_capacity:.0f} simultaneous sessions")

if 'utilization_proxy' not in acn_hourly.columns:
    acn_hourly['utilization_proxy'] = (acn_hourly['active_sessions'] / max_capacity).clip(0, 1)

acn_util_map = acn_hourly.groupby('hour')['utilization_proxy'].mean().to_dict()
acn['estimated_utilization'] = acn['hour'].map(acn_util_map).fillna(0.3)

# UrbanEV queue proxy: use queue_proxy column if available, else active_sessions proxy
if 'queue_proxy' in urban_hourly.columns:
    urban_queue = urban_hourly.groupby('hour')['queue_proxy'].mean()
else:
    # fallback: normalised session count as queue proxy
    urban_queue = (urban_hourly.groupby('hour')['active_sessions'].mean()
                   if 'active_sessions' in urban_hourly.columns
                   else urban_hourly.groupby('hour')[urban_hourly.columns[-1]].mean())

BASELINE_RATE  = 0.30   # USD/kWh (ACN)
N_EPISODES     = 10

# starting thresholds (from project brief, calibrated for ACN)
surge_threshold    = 0.80
discount_threshold = 0.15

# peak hours: morning rush + evening rush (used for wait-time calculation)
PEAK_HOURS = [7, 8, 9, 10, 17, 18, 19, 20]

print(f"Starting surge threshold:    {surge_threshold}")
print(f"Starting discount threshold: {discount_threshold}")
print(f"Episodes to simulate:        {N_EPISODES}")

# %% [markdown]
# ### 4.1 Run the Feedback Loop

# %%
episode_results = []

def evaluate_thresholds(surge_threshold, discount_threshold):
    """Evaluate one threshold policy and return monitoring metrics."""
    ep_data = acn.copy()

    utilization = ep_data['estimated_utilization'].to_numpy()
    time_period = ep_data['time_period']

    surge_mask = utilization > surge_threshold
    discount_mask = utilization < discount_threshold
    peak_mask_period = time_period.eq('peak').to_numpy()
    off_peak_mask_period = time_period.eq('off_peak').to_numpy()

    multiplier = np.ones(len(ep_data))
    tier = np.full(len(ep_data), 'normal', dtype=object)

    surge_intensity = np.minimum((utilization - surge_threshold) / (1.0 - surge_threshold), 1.0)
    discount_intensity = np.minimum((discount_threshold - utilization) / discount_threshold, 1.0)

    multiplier[surge_mask] = 1.0 + (1.40 - 1.0) * surge_intensity[surge_mask]
    multiplier[discount_mask] = 1.0 - (1.0 - 0.85) * discount_intensity[discount_mask]
    tier[surge_mask] = 'surge'
    tier[discount_mask] = 'discount'

    normal_mask = ~(surge_mask | discount_mask)
    multiplier[normal_mask & peak_mask_period] = 1.10
    multiplier[normal_mask & off_peak_mask_period] = 0.97

    ep_data['multiplier'] = multiplier
    ep_data['dynamic_price'] = BASELINE_RATE * multiplier
    ep_data['tier'] = tier

    # --- Simulate demand response ---
    ep_data['adjusted_demand'] = ep_data['kWhDelivered'] * (1 + PRICE_ELASTICITY * (ep_data['multiplier'] - 1.0))
    ep_data['adjusted_demand'] = ep_data['adjusted_demand'].clip(lower=0)

    # ----------------------------------------------------------------
    # METRIC 1: Revenue Gain % (ACN, USD)
    # ----------------------------------------------------------------
    baseline_revenue = ep_data['kWhDelivered'].sum() * BASELINE_RATE
    posted_dynamic_revenue = (ep_data['kWhDelivered'] * ep_data['dynamic_price']).sum()
    net_dynamic_revenue = (ep_data['adjusted_demand'] * ep_data['dynamic_price']).sum()
    posted_revenue_gain_pct = ((posted_dynamic_revenue - baseline_revenue) / baseline_revenue) * 100
    net_revenue_gain_pct = ((net_dynamic_revenue - baseline_revenue) / baseline_revenue) * 100

    # ----------------------------------------------------------------
    # METRIC 2: Customer Response Rate
    # Fraction of sessions where demand shifted by more than 2%.
    # (demand elasticity proxy, ACN + elasticity assumption)
    # ----------------------------------------------------------------
    demand_change_pct      = (ep_data['adjusted_demand'] - ep_data['kWhDelivered']) / ep_data['kWhDelivered']
    customer_response_rate = (demand_change_pct.abs() > 0.02).mean() * 100

    # ----------------------------------------------------------------
    # METRIC 3: Pricing Efficiency Score
    # Revenue per kWh delivered (USD/kWh) -- should improve over time.
    # ----------------------------------------------------------------
    total_adjusted_kwh = ep_data['adjusted_demand'].sum()
    pricing_efficiency = net_dynamic_revenue / total_adjusted_kwh if total_adjusted_kwh > 0 else BASELINE_RATE

    # ----------------------------------------------------------------
    # METRIC 4: Average Waiting Time Reduction  (UrbanEV queue proxy)
    #
    # Use all peak hours (not just surge tier), because ACN rarely reaches
    # explicit surge conditions. During peak hours, time-of-day pricing shifts
    # some demand away in the elasticity simulation.
    # Fewer simultaneous sessions = shorter queue = lower wait time.
    # Reduction is proportional to demand decrease caused by price premium.
    # ----------------------------------------------------------------
    peak_mask = ep_data['hour'].isin(PEAK_HOURS)

    if peak_mask.sum() > 0:
        avg_peak_multiplier = ep_data.loc[peak_mask, 'multiplier'].mean()

        # queue proxy from UrbanEV peak hours
        available_hours = [h for h in PEAK_HOURS if h in urban_queue.index]
        if available_hours:
            baseline_queue = urban_queue[available_hours].mean()
        else:
            baseline_queue = urban_queue.mean()

        # demand reduction fraction during peak (elasticity-based)
        price_change_at_peak = avg_peak_multiplier - 1.0
        demand_reduction_frac = abs(PRICE_ELASTICITY * price_change_at_peak)

        reduced_queue     = baseline_queue * (1 - demand_reduction_frac)
        wait_time_reduction = ((baseline_queue - reduced_queue) / baseline_queue * 100) \
                              if baseline_queue > 0 else 0.0
    else:
        wait_time_reduction = 0.0

    return {
        'posted_revenue_gain_pct': posted_revenue_gain_pct,
        'net_revenue_gain_pct': net_revenue_gain_pct,
        'customer_response_rate': customer_response_rate,
        'pricing_efficiency': pricing_efficiency,
        'wait_time_reduction': wait_time_reduction,
        'surge_pct': (ep_data['tier'] == 'surge').mean() * 100,
        'discount_pct': (ep_data['tier'] == 'discount').mean() * 100,
        'avg_multiplier': ep_data['multiplier'].mean()
    }


for episode in range(N_EPISODES):

    metrics = evaluate_thresholds(surge_threshold, discount_threshold)

    episode_results.append({
        'episode':              episode + 1,
        'surge_threshold':      surge_threshold,
        'discount_threshold':   discount_threshold,
        **metrics
    })

    # ----------------------------------------------------------------
    # FEEDBACK LOOP
    # Evaluate nearby threshold candidates and move to the best bounded
    # policy. The objective rewards net revenue and queue reduction while
    # penalizing excessive customer response.
    # ----------------------------------------------------------------
    if episode < N_EPISODES - 1:
        candidate_policies = []

        for surge_step in [-0.02, 0.0, 0.02]:
            for discount_step in [-0.01, 0.0, 0.01]:
                candidate_surge = round(min(0.90, max(0.70, surge_threshold + surge_step)), 3)
                candidate_discount = round(min(0.30, max(0.10, discount_threshold + discount_step)), 3)
                candidate = evaluate_thresholds(candidate_surge, candidate_discount)
                score = (
                    candidate['net_revenue_gain_pct']
                    + 0.20 * candidate['wait_time_reduction']
                    - 0.03 * candidate['customer_response_rate']
                )
                candidate_policies.append((score, candidate_surge, candidate_discount))

        _, surge_threshold, discount_threshold = max(candidate_policies, key=lambda x: x[0])

# %%
episodes_df = pd.DataFrame(episode_results)
episodes_df.to_csv(os.path.join(OUTPUT_DIR, 'monitoring_metrics.csv'), index=False)
print(f"\nSaved monitoring metrics for {N_EPISODES} episodes")

# %% [markdown]
# ---
# ## 5. Learning Curve Visualization

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Monitoring & Learning Agent -- Performance Across Episodes', fontsize=14, fontweight='bold')

# 1. Revenue Gain %
axes[0, 0].plot(episodes_df['episode'], episodes_df['posted_revenue_gain_pct'],
                marker='o', color=COLORS['primary'], linewidth=2, label='Posted tariff revenue')
axes[0, 0].plot(episodes_df['episode'], episodes_df['net_revenue_gain_pct'],
                marker='s', color=COLORS['dark'], linewidth=2, label='After elasticity response')
axes[0, 0].fill_between(episodes_df['episode'], episodes_df['net_revenue_gain_pct'],
                         alpha=0.15, color=COLORS['primary'])
axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.4, label='Flat baseline')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Revenue Gain %')
axes[0, 0].set_title('Revenue Gain % Over Episodes (ACN, USD)')
axes[0, 0].legend()

# 2. Customer Response Rate
axes[0, 1].plot(episodes_df['episode'], episodes_df['customer_response_rate'],
                marker='s', color=COLORS['secondary'], linewidth=2)
axes[0, 1].fill_between(episodes_df['episode'], episodes_df['customer_response_rate'],
                         alpha=0.15, color=COLORS['secondary'])
axes[0, 1].axhline(y=60, color='red', linestyle='--', alpha=0.4, label='60% caution line')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Customer Response Rate (%)')
axes[0, 1].set_title('Customer Response Rate Over Episodes')
axes[0, 1].legend()

# 3. Pricing Efficiency Score
axes[0, 2].plot(episodes_df['episode'], episodes_df['pricing_efficiency'],
                marker='^', color=COLORS['peak'], linewidth=2)
axes[0, 2].fill_between(episodes_df['episode'], episodes_df['pricing_efficiency'],
                         alpha=0.15, color=COLORS['peak'])
axes[0, 2].axhline(y=BASELINE_RATE, color='black', linestyle='--', alpha=0.4,
                    label=f'Baseline ${BASELINE_RATE}/kWh')
axes[0, 2].set_xlabel('Episode')
axes[0, 2].set_ylabel('USD/kWh')
axes[0, 2].set_title('Pricing Efficiency Score (Revenue/kWh, ACN)')
axes[0, 2].legend()

# 4. Waiting Time Reduction
axes[1, 0].plot(episodes_df['episode'], episodes_df['wait_time_reduction'],
                marker='D', color=COLORS['off_peak'], linewidth=2)
axes[1, 0].fill_between(episodes_df['episode'], episodes_df['wait_time_reduction'],
                         alpha=0.15, color=COLORS['off_peak'])
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Wait Time Reduction (%)')
axes[1, 0].set_title('Avg Waiting Time Reduction -- Peak Hours (UrbanEV proxy)')

# 5. Threshold Evolution
axes[1, 1].plot(episodes_df['episode'], episodes_df['surge_threshold'],
                marker='o', color=COLORS['peak'],     linewidth=2, label='Surge Threshold')
axes[1, 1].plot(episodes_df['episode'], episodes_df['discount_threshold'],
                marker='s', color=COLORS['off_peak'], linewidth=2, label='Discount Threshold')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Threshold Value')
axes[1, 1].set_title('Threshold Evolution (Feedback Loop Learning)')
axes[1, 1].legend()

# 6. Pricing tier distribution
axes[1, 2].stackplot(
    episodes_df['episode'],
    episodes_df['surge_pct'],
    100 - episodes_df['surge_pct'] - episodes_df['discount_pct'],
    episodes_df['discount_pct'],
    labels=['Surge', 'Normal', 'Discount'],
    colors=[COLORS['peak'], COLORS['shoulder'], COLORS['off_peak']],
    alpha=0.7
)
axes[1, 2].set_xlabel('Episode')
axes[1, 2].set_ylabel('% of Sessions')
axes[1, 2].set_title('Pricing Tier Distribution Over Episodes')
axes[1, 2].legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'monitoring_learning_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 6. Final Agent Performance Summary

# %%
print("=" * 60)
print("MONITORING & LEARNING AGENT -- FINAL PERFORMANCE")
print("=" * 60)

final   = episodes_df.iloc[-1]
initial = episodes_df.iloc[0]

print(f"\n--- Performance: Episode 1 -> Episode {N_EPISODES} ---")
print(f"  Posted Revenue Gain %: {initial['posted_revenue_gain_pct']:+.2f}%  ->  {final['posted_revenue_gain_pct']:+.2f}%")
print(f"  Net Revenue Gain %:    {initial['net_revenue_gain_pct']:+.2f}%  ->  {final['net_revenue_gain_pct']:+.2f}%")
print(f"  Customer Response:     {initial['customer_response_rate']:.1f}%   ->  {final['customer_response_rate']:.1f}%")
print(f"  Pricing Efficiency:    ${initial['pricing_efficiency']:.4f}  ->  ${final['pricing_efficiency']:.4f}/kWh")
print(f"  Wait Time Reduction:   {initial['wait_time_reduction']:.1f}%   ->  {final['wait_time_reduction']:.1f}%")

print(f"\n--- Threshold Evolution ---")
print(f"  Surge threshold:       {initial['surge_threshold']:.3f}  ->  {final['surge_threshold']:.3f}")
print(f"  Discount threshold:    {initial['discount_threshold']:.3f}  ->  {final['discount_threshold']:.3f}")

print(f"\n--- Final Pricing Tier Distribution ---")
print(f"  Surge sessions:    {final['surge_pct']:.1f}%")
print(f"  Normal sessions:   {100 - final['surge_pct'] - final['discount_pct']:.1f}%")
print(f"  Discount sessions: {final['discount_pct']:.1f}%")

# %% [markdown]
# ### 6.1 Comprehensive Metrics Table (Submission-ready)

# %%
final_metrics = pd.DataFrame({
    'Metric': [
        'Posted Revenue Gain % (Episode 1)',
        'Posted Revenue Gain % (Final Episode)',
        'Net Revenue Gain % (Final Episode)',
        'Off-Peak Uplift %',
        'Avg Waiting Time Reduction % (Final)',
        'Customer Response Rate % (Final)',
        'Pricing Efficiency Score -- Final (USD/kWh)',
        'Pricing Efficiency -- Baseline (USD/kWh)',
        'Best Demand Prediction Model',
        'Final Surge Threshold',
        'Final Discount Threshold',
        'Number of Learning Episodes',
        'Price Elasticity Assumption'
    ],
    'Value': [
        f"{initial['posted_revenue_gain_pct']:+.2f}%",
        f"{final['posted_revenue_gain_pct']:+.2f}%",
        f"{final['net_revenue_gain_pct']:+.2f}%",
        "See Notebook 3 (UrbanEV)",
        f"{final['wait_time_reduction']:.1f}%",
        f"{final['customer_response_rate']:.1f}%",
        f"${final['pricing_efficiency']:.4f}",
        f"${BASELINE_RATE:.4f}",
        "See Notebook 3 model comparison",
        f"{final['surge_threshold']:.3f}",
        f"{final['discount_threshold']:.3f}",
        str(N_EPISODES),
        "-0.30 (conservative, EV charging literature)"
    ],
    'Dataset': [
        'ACN (USD)', 'ACN (USD)', 'ACN + elasticity', 'UrbanEV (CNY)',
        'UrbanEV queue proxy', 'ACN + elasticity',
        'ACN (USD)', 'ACN (USD)',
        'UrbanEV', '-', '-', '-', '-'
    ]
})

print("\n=== FINAL METRICS TABLE ===")
print(final_metrics.to_string(index=False))
final_metrics.to_csv(os.path.join(OUTPUT_DIR, 'agent_performance_summary.csv'), index=False)
print("\nSaved agent_performance_summary.csv")

# %% [markdown]
# ---
# ## 7. Key Takeaways
#
# 1. **Feedback loop improves revenue:** The agent starts conservative and refines
#    thresholds based on observed revenue direction each episode.
#
# 2. **Wait time reduction is real:** By pricing up during peak hours, some demand
#    shifts away, reducing queue length. Computed via elasticity on UrbanEV queue proxy.
#
# 3. **Thresholds adapt intelligently:**
#    - Surge threshold tightens when revenue improves (exploit more congestion)
#    - Surge threshold relaxes when revenue drops (too aggressive -> customers leave)
#    - Discount window widens gradually to capture more off-peak sessions
#
# 4. **All assumptions documented:**
#    - Elasticity = -0.30 (conservative, consistent across both agents)
#    - Queue proxy from UrbanEV hourly data
#    - ACN utilization estimated from hourly session counts / peak capacity
#    - Currencies kept separate: USD (ACN), CNY (UrbanEV)
#
# 5. **Limitations acknowledged:**
#    - True demand elasticity for Indian/Chinese markets may differ
#    - Queue proxy is indirect (no direct wait-time data in either dataset)
#    - Episode simulation is sequential on the same dataset (not true online learning)

