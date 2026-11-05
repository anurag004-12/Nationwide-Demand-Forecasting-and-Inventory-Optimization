import pandas as pd
from src.config import HORIZON

# The columns the models will learn from
FEATURES = [
    "lag_28", "lag_35", "lag_42", "lag_56", "lag_364",
    "rmean_7", "rmean_28", "rmean_56", "rstd_28",
    "sell_price", "price_rel", "promo", "price_chg",
    "dow", "month", "dom", "woy", "is_event", "snap",
    "store_c", "item_c",
]
CATEGORICAL = ["store_c", "item_c"]


def add_features(df):
    """Add lag, rolling, price, and calendar features."""
    df = df.copy()
    by_series = df.groupby("id")["sales"]

    # Lags: sales from N days ago.
    # IMPORTANT: every lag is >= 28 because we predict 28 days ahead.
    # A lag_7 would use data we won't have yet (data leakage).
    for lag in [28, 35, 42, 56, 364]:
        df[f"lag_{lag}"] = by_series.shift(lag)

    # Rolling averages, also shifted by 28 days to avoid leakage
    for window in [7, 28, 56]:
        df[f"rmean_{window}"] = by_series.transform(
            lambda s: s.shift(HORIZON).rolling(window).mean())
    df["rstd_28"] = by_series.transform(
        lambda s: s.shift(HORIZON).rolling(28).std())

    # Price features
    df["price_med"] = df.groupby("id")["sell_price"].transform(lambda s: s.rolling(90, min_periods=1).median())
    df["price_rel"] = df["sell_price"] / df["price_med"]       # price vs usual price
    df["promo"] = (df["price_rel"] < 0.95).astype(int)         # 1 if >5% cheaper than usual
    df["price_chg"] = df.groupby("id")["sell_price"].pct_change().fillna(0)

    # Calendar features
    df["dow"] = df["wday"]
    df["dom"] = df["date"].dt.day
    df["woy"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_event"] = df["event_name_1"].notna().astype(int)

    # Turn store and item names into numbers LightGBM can use as categories
    df["store_c"] = df["store_id"].astype("category").cat.codes
    df["item_c"] = df["item_id"].astype("category").cat.codes
    return df