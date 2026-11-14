# Architecture and data documentation

![Architecture](docs/architecture.png)

## Design decisions

| Decision | Reason |
|---|---|
| 28-day direct forecast, all lags >= 28 | Avoids leakage: no feature uses data from inside the forecast window |
| Tweedie loss for LightGBM | Sales are non-negative counts with many zeros (14% zero days in the holdout) |
| Quantile LightGBM for intervals | Gives 5/25/50/75/95% quantiles, also used as demand scenarios |
| MinT reconciliation | Makes item, store and state forecasts add up |
| Scenario-based LP in OR-Tools | Minimizes expected leftover + shortage cost under budget and capacity limits |

## Output files (`outputs/`)

| File | Columns | Used by |
|---|---|---|
| `forecasts.csv` | id, store_id, item_id, date, actual, forecast, lo, hi | App chart |
| `orders.csv` | id, store_id, price, d_q05 ... d_q95, unit_cost, h, s, order_qty | App order table |
| `policies.csv` | policy, fill_rate, leftover_units, total_cost | App policy table |
| `metrics.csv` | model, level, WAPE, RMSE | App metrics table |
| `extras.csv` | item, value (gaps, coverage, interval width) | App summary cards |