# EV Dynamic Tariff Optimization

This project studies how EV charging prices can be adjusted using demand forecasts, charger utilization, and a simple monitoring loop. The goal is not to claim a real-world causal impact, but to build a reproducible decision-support pipeline for dynamic tariff design.

The work was built for the OP'26 Analytics problem statement on Agentic AI-based Dynamic Tariff Optimization for EV charging networks.

## What This Project Does

The solution has four notebooks:

1. `01_data_preprocessing.ipynb`
   - Cleans and prepares the ACN and UrbanEV datasets.
   - Builds hourly demand/utilization tables used by later notebooks.

2. `02_eda.ipynb`
   - Studies hourly, weekday/weekend, zone-level, and price-demand patterns.
   - Produces the plots used to justify the tariff rules.

3. `03_demand_tariff_agent.ipynb`
   - Trains demand prediction models on UrbanEV data.
   - Converts predicted utilization into dynamic tariff decisions.
   - Evaluates revenue gain, utilization change, and off-peak uplift.

4. `04_monitoring_agent.ipynb`
   - Simulates a monitoring and learning loop.
   - Tracks revenue, customer response, pricing efficiency, and waiting-time proxy over 10 episodes.



## Dataset Use

Both datasets are used because they support different parts of the problem:

- UrbanEV/ST-EVCDP is used for utilization forecasting, spatial/temporal demand analysis, off-peak uplift, and the congestion/waiting-time proxy.
- ACN-Data is used for session-level revenue, customer response simulation, and pricing efficiency.

The datasets are not merged into one geography or currency. UrbanEV is used mainly for utilization behavior, while ACN revenue is reported in USD.

## Current Results

The latest generated outputs show:

| Area | Metric | Result |
|---|---:|---:|
| Demand prediction | Best model | XGBoost |
| Demand prediction | RMSE | 0.0505 |
| Demand prediction | MAE | 0.0299 |
| Demand prediction | R2 score | 0.9183 |
| Tariff pricing | ACN revenue gain vs flat baseline | +4.62% |
| Tariff pricing | UrbanEV off-peak uplift | +2.7% |
| Monitoring loop | Final net revenue gain after elasticity | +4.09% |
| Monitoring loop | Final customer response / demand retained | 98.2% |
| Monitoring loop | Final pricing efficiency | $0.3181/kWh |
| Monitoring loop | Final waiting-time proxy reduction | 2.01% |

These are simulation results based on the assumptions in the notebooks.

## Folder Structure

```text
Socbiz_project/
  data/
    acn/              raw ACN data
    urbanev/          raw UrbanEV data
    processed/        generated locally after running Notebook 1
  notebooks/
    01_data_preprocessing.ipynb
    02_eda.ipynb
    03_demand_tariff_agent.ipynb
    04_monitoring_agent.ipynb
  outputs/
    summary CSV results and plots
  presentation/
    final deck files
  README.md
```

## How To Run

Run the notebooks in this order:

1. `notebooks/01_data_preprocessing.ipynb`
2. `notebooks/02_eda.ipynb`
3. `notebooks/03_demand_tariff_agent.ipynb`
4. `notebooks/04_monitoring_agent.ipynb`

Do not run an `.ipynb` file with `python notebook.ipynb`. Open the notebooks in Jupyter, VS Code, or Colab and run the cells from top to bottom.

## Important Output Files

The most useful output files are:

- `outputs/model_comparison.csv`
- `outputs/agent1_agent2_kpi.csv`
- `outputs/revenue_comparison.csv`
- `outputs/charger_utilization_kpi.csv`
- `outputs/monitoring_metrics.csv`
- `outputs/agent3_kpi.csv`
- `outputs/agent_performance_summary.csv`
- supporting `.png` plots in `outputs/`

Large generated files such as `demand_predictions.csv`, `dynamic_tariffs_urbanev.csv`, `dynamic_tariffs_acn.csv`, and the trained model `.pkl` are intentionally ignored for GitHub. They are recreated by running the notebooks.

## Assumptions 

- Waiting time is not directly observed, so high-utilization UrbanEV periods are used as a queue/waiting-time proxy.
- Customer response is simulated using a conservative price elasticity assumption of `-0.30`.
- Results should be read as modeled decision-support outcomes, not causal proof.
- A real deployment would need live testing, fairness checks, user acceptance analysis, and local tariff regulation review.

