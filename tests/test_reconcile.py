import numpy as np
import pandas as pd

from src.config import STATE
from src.reconcile import reconcile


def toy_data():
    rng = np.random.default_rng(1)
    hist_dates = pd.date_range("2016-01-01", periods=100)
    val_dates = pd.date_range(hist_dates[-1] + pd.Timedelta(days=1), periods=28)
    items = {"a_CA_1": "CA_1", "b_CA_1": "CA_1", "a_CA_2": "CA_2", "b_CA_2": "CA_2"}

    history = pd.DataFrame([
        {"id": i, "store_id": s, "date": d, "sales": 50 + rng.normal(0, 5)}
        for i, s in items.items() for d in hist_dates])
    valid = pd.DataFrame([
        {"id": i, "store_id": s, "date": d, "lgb": 50 + rng.normal(0, 3)}
        for i, s in items.items() for d in val_dates])

    # Prophet-style forecasts for the upper levels: deliberately inconsistent with the items
    prophet_fc = {
        "CA_1": pd.Series(115 + rng.normal(0, 3, 28), index=val_dates),
        "CA_2": pd.Series(95 + rng.normal(0, 3, 28), index=val_dates),
        STATE: pd.Series(250 + rng.normal(0, 3, 28), index=val_dates),
    }
    return history, valid, prophet_fc


def test_reconciliation_makes_forecasts_add_up():
    history, valid, prophet_fc = toy_data()
    rec, gap_before, gap_after = reconcile(history, valid, prophet_fc)

    assert gap_before > 1.0            # the inputs really disagree
    assert gap_after < 1e-6            # after MinT they add up


def test_reconciled_output_is_complete_and_non_negative():
    history, valid, prophet_fc = toy_data()
    rec, _, _ = reconcile(history, valid, prophet_fc)

    assert len(rec) == 4 * 28
    assert rec["lgb_rec"].notna().all()
    assert (rec["lgb_rec"] >= 0).all()