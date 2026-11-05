import pandas as pd
from src.config import RAW, STATE, DEPARTMENT, TOP_N_ITEMS


def load_data():
    """Read the 3 M5 files, keep a small subset, return one long table."""
    calendar = pd.read_csv(RAW / "calendar.csv")
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    prices = pd.read_csv(RAW / "sell_prices.csv")

    # 1. Keep only one state and one department (so it runs on a laptop)
    sales = sales[(sales["state_id"] == STATE) & (sales["dept_id"] == DEPARTMENT)]

    # 2. Keep the top N items by total sales
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    total_per_item = sales.groupby("item_id")[day_cols].sum().sum(axis=1)
    top_items = total_per_item.nlargest(TOP_N_ITEMS).index
    sales = sales[sales["item_id"].isin(top_items)]

    # 3. Wide -> long: one row per item-store-day
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df = sales.melt(id_vars=id_cols, value_vars=day_cols,
                    var_name="d", value_name="sales")

    # 4. Add calendar info (date, weekday, events, SNAP days)
    cal_cols = ["d", "date", "wm_yr_wk", "wday", "month",
                "event_name_1", f"snap_{STATE}"]
    df = df.merge(calendar[cal_cols], on="d", how="left")
    df = df.rename(columns={f"snap_{STATE}": "snap"})

    # 5. Add prices (matched on store + item + week)
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # 6. Clean up
    df["date"] = pd.to_datetime(df["date"])
    df["t"] = df["d"].str[2:].astype(int)            # "d_1914" -> 1914
    df = df.dropna(subset=["sell_price"])            # price is missing before an item launches
    df = df.sort_values(["id", "t"]).reset_index(drop=True)
    return df