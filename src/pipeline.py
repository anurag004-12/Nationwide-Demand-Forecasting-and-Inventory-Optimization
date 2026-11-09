import numpy as np
import pandas as pd

from src.config import HORIZON, OUTPUTS, PROCESSED, CAPACITY_FACTOR
from src.data import load_data
from src.features import add_features, FEATURES
from src.metrics import score, coverage
from src.models.baselines import prophet_forecasts
from src.models.lgbm import train_point_model, train_quantile_models
from src.reconcile import reconcile
from src.optimize import (make_item_table, optimise_orders,
                          weekly_actuals, evaluate_policy)


def main():
    OUTPUTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. Data and features
    print("Loading data and building features...")
    df = add_features(load_data())
    df.to_parquet(PROCESSED / "features.parquet", index=False)   # TFT on Colab reads this

    # 2. Time-based split: the last 28 days are the test period
    cut = df["t"].max() - HORIZON
    history = df[df["t"] <= cut]
    train = history.dropna(subset=["lag_364"])
    valid = df[df["t"] > cut].copy()
    results = []                                                  # (model, level, WAPE, RMSE)

    def add_result(name, level, y, p):
        w, r = score(y, p)
        results.append({"model": name, "level": level, "WAPE": w, "RMSE": r})

    # 3. Naive baseline: same weekday, 28 days ago
    valid["naive"] = valid["lag_28"].fillna(0)
    add_result("Naive", "item-store", valid["sales"], valid["naive"])

    # 4. Prophet (store and state totals only)
    print("Running Prophet...")
    prophet_fc = prophet_forecasts(history)

    # 5. LightGBM point model
    print("Training LightGBM...")
    model = train_point_model(train)
    valid["lgb"] = model.predict(valid[FEATURES]).clip(0)
    add_result("LightGBM", "item-store", valid["sales"], valid["lgb"])

    # 6. Reconciliation
    print("Reconciling...")
    rec_long, gap_before, gap_after = reconcile(history, valid, prophet_fc)
    valid = valid.merge(rec_long, on=["id", "date"], how="left")
    add_result("LightGBM + MinT", "item-store", valid["sales"], valid["lgb_rec"])
    print(f"Item-sum vs state gap: before={gap_before:.2f}, after={gap_after:.4f}")

    # 7. Store-level comparison (the only level where Prophet can be scored)
    def by_store(col):
        return valid.groupby(["date", "store_id"])[col].sum().unstack()

    actual_store = by_store("sales")
    prophet_store = pd.DataFrame({s: prophet_fc[s] for s in actual_store.columns})
    prophet_store = prophet_store.reindex(actual_store.index)
    for name, col in [("Naive", "naive"), ("LightGBM", "lgb"), ("LightGBM + MinT", "lgb_rec")]:
        add_result(name, "store", actual_store.values, by_store(col).values)
    add_result("Prophet", "store", actual_store.values, prophet_store.values)

    # 8. Prediction intervals (quantile models)
    print("Training quantile models...")
    q_models = train_quantile_models(train)
    for q, m in q_models.items():
        valid[f"q{round(q * 100):02d}"] = m.predict(valid[FEATURES]).clip(0)
    qcols = ["q05", "q25", "q50", "q75", "q95"]
    valid[qcols] = np.sort(valid[qcols].values, axis=1)           # avoid crossing quantiles
    cov90 = coverage(valid["sales"], valid["q05"], valid["q95"])
    cov50 = coverage(valid["sales"], valid["q25"], valid["q75"])
    width = (valid["q95"] - valid["q05"]).mean()
    print(f"90% coverage {cov90:.3f} | 50% coverage {cov50:.3f} | width {width:.2f}")

    # 9. TFT results (only if trained on Colab and copied into outputs/)
    tft_file = OUTPUTS / "tft_preds.csv"
    if tft_file.exists():
        tft = pd.read_csv(tft_file, parse_dates=["date"])
        merged = valid.merge(tft, on=["id", "date"])
        add_result("TFT", "item-store", merged["sales"], merged["tft"])

    # 10. Order optimisation and policy comparison
    print("Optimising orders...")
    items = make_item_table(valid)
    actual = weekly_actuals(valid)
    capacity = (items.groupby("store_id")["d_q50"].sum() * CAPACITY_FACTOR).to_dict()

    orders = optimise_orders(items, store_capacity=capacity)
    policies = pd.DataFrame([
        evaluate_policy(items.assign(order_qty=items["d_q50"].round()), actual,
                        "Order the median forecast"),
        evaluate_policy(items.assign(order_qty=np.floor(items["d_q50"] * CAPACITY_FACTOR)), actual,
                        f"Even {CAPACITY_FACTOR}x median (same capacity)"),
        evaluate_policy(orders, actual, "Optimized, store capacity"),
        evaluate_policy(optimise_orders(items), actual, "Optimized, no limit"),
    ])
    print(policies)

    # 11. Save everything the app needs
    forecasts = valid[["id", "store_id", "item_id", "date", "sales", "lgb_rec", "q05", "q95"]].copy()
    forecasts = forecasts.rename(columns={"sales": "actual", "lgb_rec": "forecast",
                                          "q05": "lo", "q95": "hi"})
    # Reconciled forecasts can fall slightly outside the unreconciled interval; widen to include them
    forecasts["lo"] = np.minimum(forecasts["lo"], forecasts["forecast"])
    forecasts["hi"] = np.maximum(forecasts["hi"], forecasts["forecast"])
    forecasts.to_csv(OUTPUTS / "forecasts.csv", index=False)

    orders.round(2).to_csv(OUTPUTS / "orders.csv", index=False)
    policies.to_csv(OUTPUTS / "policies.csv", index=False)

    metrics = pd.DataFrame(results)
    metrics.to_csv(OUTPUTS / "metrics.csv", index=False)

    extras = pd.DataFrame({
        "item": ["gap_before", "gap_after", "coverage_90", "coverage_50", "avg_width_90"],
        "value": [gap_before, gap_after, cov90, cov50, width],
    })
    extras.round(4).to_csv(OUTPUTS / "extras.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()