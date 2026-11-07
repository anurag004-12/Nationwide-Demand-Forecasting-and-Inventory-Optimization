import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import MinTReconciliator
from src.config import STATE


def make_wide(d, col):
    """Build a table with one column per series at every level:
    items (bottom), stores (middle), state (top)."""
    bottom = d.pivot(index="date", columns="id", values=col).fillna(0)
    store = d.groupby(["date", "store_id"])[col].sum().unstack()
    state = d.groupby("date")[col].sum().rename(STATE)
    return pd.concat([bottom, store, state], axis=1).astype(float)


def reconcile(history, valid, prophet_fc):
    """Make item, store, and state forecasts add up (MinT reconciliation).
    Returns the reconciled item forecasts and the 'gap' before and after."""
    # Which stores/state does each item-series belong to?
    pairs = history[["id", "store_id"]].drop_duplicates()
    # Each item-store series sits under its store, and under the state
    hierarchy = {i: [s, STATE] for i, s in pairs.itertuples(index=False)}

# Each store sits under the state (this was the missing part)
    for store in pairs["store_id"].unique():
        hierarchy[store] = [STATE]

    # Actual history (used to learn how errors are related across levels)
    train_wide = make_wide(history, "sales")

    # Base forecasts: LightGBM for items, Prophet for stores and state.
    # They come from different models, so they do NOT add up yet.
    bottom = valid.pivot(index="date", columns="id", values="lgb").fillna(0)
    stores = pd.DataFrame({k: v for k, v in prophet_fc.items() if k != STATE})
    state = prophet_fc[STATE].rename(STATE)
    base = pd.concat([bottom, stores, state], axis=1)
    base.index.name = "date"
    base = base[train_wide.columns].astype(float)

    train_ts = TimeSeries.from_dataframe(train_wide, freq="D", hierarchy=hierarchy)
    pred_ts = TimeSeries.from_dataframe(base, freq="D", hierarchy=hierarchy)

    reconciler = MinTReconciliator(method="wls_val")
    reconciler.fit(train_ts)
    raw = reconciler.transform(pred_ts).to_dataframe()
    fixed = raw.clip(lower=0)

    item_cols = list(bottom.columns)
    #print("gap without clipping:", (raw[item_cols].sum(axis=1) - raw[STATE]).abs().mean())
    gap_before = (base[item_cols].sum(axis=1) - base[STATE]).abs().mean()
    gap_after = (fixed[item_cols].sum(axis=1) - fixed[STATE]).abs().mean()

    long = fixed[item_cols].reset_index()
    long = long.rename(columns={long.columns[0]: "date"})
    long = long.melt(id_vars="date", var_name="id", value_name="lgb_rec")
    return long, gap_before, gap_after