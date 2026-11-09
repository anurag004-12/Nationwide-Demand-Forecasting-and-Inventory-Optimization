import numpy as np
import pandas as pd

from src.config import HORIZON
from src.features import FEATURES, add_features


def make_df(n_days=120, n_series=2):
    rng = np.random.default_rng(0)
    frames = []
    for k in range(n_series):
        dates = pd.date_range("2015-01-01", periods=n_days)
        frames.append(pd.DataFrame({
            "id": f"item{k}_CA_1", "item_id": f"item{k}", "store_id": "CA_1",
            "t": np.arange(1, n_days + 1), "date": dates,
            "sales": rng.integers(0, 30, size=n_days),
            "sell_price": 2.0, "wday": dates.dayofweek + 1, "month": dates.month,
            "event_name_1": None, "snap": 0,
        }))
    return pd.concat(frames, ignore_index=True)


def test_lag_names_are_at_least_horizon():
    lag_cols = [c for c in FEATURES if c.startswith("lag_")]
    assert all(int(c.split("_")[1]) >= HORIZON for c in lag_cols)


def test_lag_28_is_sales_28_days_earlier():
    out = add_features(make_df())
    s = out[out["id"] == "item0_CA_1"].reset_index(drop=True)
    assert s["lag_28"].iloc[:28].isna().all()
    assert (s["lag_28"].iloc[28:].values == s["sales"].iloc[:-28].values).all()


def test_lags_do_not_leak_across_series():
    out = add_features(make_df())
    second = out[out["id"] == "item1_CA_1"]
    assert second["lag_28"].iloc[:28].isna().all()     # must not pick up series 0's sales


def test_future_sales_cannot_change_earlier_features():
    """Change one day's sales. No lag or rolling feature may change in the 27 days after it,
    because every one of them is shifted by at least 28 days."""
    df = make_df()
    before = add_features(df)

    changed = df.copy()
    mask = (changed["id"] == "item0_CA_1") & (changed["t"] == 60)
    changed.loc[mask, "sales"] = 10_000
    after = add_features(changed)

    rows = (before["id"] == "item0_CA_1") & (before["t"] < 60 + HORIZON)
    for col in ["lag_28", "lag_35", "lag_42", "lag_56",
                "rmean_7", "rmean_28", "rmean_56", "rstd_28"]:
        assert np.allclose(before.loc[rows, col], after.loc[rows, col], equal_nan=True), col