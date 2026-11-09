# Nationwide Demand Forecasting and Inventory Optimization

Forecasts daily demand for Walmart items 28 days ahead, produces prediction intervals,
makes item, store and state forecasts consistent, and turns the forecasts into order
quantities with OR-Tools. Results are shown in a Plotly Dash app.

**Dataset:** M5 Forecasting (Walmart), Kaggle.
**Scope:** California (4 stores), department FOODS_3, top 50 items by sales = 200 item-store series.
Chosen so the full pipeline runs on a laptop.

## Architecture

```mermaid
flowchart LR
    A[M5 CSVs<br/>calendar, sales, prices] --> B[data.py<br/>subset, melt, merge]
    B --> C[features.py<br/>lags >= 28, rolling, price, calendar]

    subgraph Models
        N[Naive]
        L[LightGBM point]
        Q[LightGBM quantiles<br/>5/25/50/75/95%]
        P[Prophet<br/>store and state totals]
        T[TFT<br/>Colab GPU, optional]
    end

    B --> P
    C --> N
    C --> L
    C --> Q
    C -. features.parquet .-> T

    L --> R[reconcile.py<br/>MinT]
    P --> R
    Q --> O[optimize.py<br/>OR-Tools]

    Models --> E[Evaluation<br/>WAPE, RMSE, coverage]
    T -. tft_preds.csv .-> E

    R --> G[outputs/*.csv]
    O --> G
    Q --> G
    E --> G
    G --> D[Dash app]
```

## Results

Holdout: last 28 days of the data (days 1914 to 1941). Lower is better.

**Item-store level (200 series)**

| Model | WAPE | RMSE |
|---|---|---|
| Naive (same weekday, 28 days ago) | 0.4947 | 10.24 |
| LightGBM | 0.3830 | 7.46 |
| LightGBM + MinT reconciliation | 0.3827 | 7.41 |

**Store level (sum of items per store; the only level Prophet is scored at)**

| Model | WAPE | RMSE |
|---|---|---|
| Naive | 0.1026 | 91.08 |
| LightGBM | 0.0952 | 85.91 |
| LightGBM + MinT | 0.0975 | 82.16 |
| Prophet | 0.1187 | 95.02 |

**Reconciliation:** the average gap between the sum of item forecasts and the state forecast
fell from 92.2 to about 0 units/day. Clipping negative forecasts to zero leaves 1.35 units/day.
Accuracy barely changes, so the benefit is consistency.

**Prediction intervals:** the 90% interval covers 92.6% of actual values, the 50% interval covers
55.3%. Average 90% width: 23.98 units.

**Ordering policies**, scored against the real first week of the holdout:

| Policy | Fill rate | Leftover units | Total cost |
|---|---|---|---|
| Order the median forecast | 0.875 | 1,973 | 2,283 |
| Even 1.1x median (same capacity) | 0.919 | 2,833 | 1,634 |
| Optimized, store capacity | 0.897 | 3,404 | 1,469 |
| Optimized, no limit | 0.998 | 18,413 | 1,143 |

## Assumptions (all inputs, change them in `src/optimize.py` and `src/config.py`)

- Unit cost = 70% of price, so margin = 30%.
- Holding or waste cost per leftover unit = 5% of unit cost.
- Shortage cost per missed unit = 1.5 x margin.
- Order horizon = 7 days. Shelf capacity per store = 1.1 x median weekly demand.
- Weekly demand scenarios are built from the 5/25/50/75/95% quantiles with weights 0.1/0.2/0.4/0.2/0.1.

## Limitations

- One 28-day holdout window, so results are single estimates, not averages over several periods.
- Costs are assumed, not real. The optimizer is judged with the same costs it minimizes.
- Weekly demand quantiles are obtained by summing daily quantiles. This overstates the weekly
  spread and inflates the unconstrained order quantities.
- Prophet only forecasts store and state totals, and ignores prices and promotions.
- No on-hand inventory or lead times are modelled.
- Interval quantiles come from unreconciled models, so intervals are not coherent across levels.

## Setup

Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Download the M5 data from Kaggle (M5 Forecasting - Accuracy) and place these in `data/raw/`:
`calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`.

## Run

```bash
python -m src.pipeline          # trains models, writes outputs/*.csv (several minutes)
python app/app.py               # opens at http://127.0.0.1:8050
python -m pytest -v             # 13 tests
```

### Optional: TFT on Colab (GPU)

1. Run the pipeline once locally to create `data/processed/features.parquet`.
2. In Colab, clone this repo, copy the parquet in, and run `python -m src.models.tft`.
3. Copy the resulting `outputs/tft_preds.csv` back and rerun the pipeline. A TFT row then appears
   in `outputs/metrics.csv`.

## Project layout

```
src/            data, features, models, reconciliation, optimizer, pipeline
app/            Dash app
outputs/        forecasts, orders, policies, metrics (read by the app)
tests/          metrics, leakage, reconciliation and optimizer tests
notebooks/      exploration
docs/           architecture, results, reflection
```

