import numpy as np
import pandas as pd

from src.optimize import optimise_orders


def toy_items(n=2, store="CA_1"):
    return pd.DataFrame({
        "id": [f"item{i}" for i in range(n)], "store_id": [store] * n, "price": [10.0] * n,
        "d_q05": [5.0] * n, "d_q25": [8.0] * n, "d_q50": [10.0] * n,
        "d_q75": [12.0] * n, "d_q95": [15.0] * n,
        "unit_cost": [7.0] * n, "h": [0.35] * n, "s": [4.5] * n,
    })


def test_budget_respected_and_orders_non_negative():
    result = optimise_orders(toy_items(), budget=100)
    assert (result["order_qty"] >= 0).all()
    assert (result["order_qty"] * result["unit_cost"]).sum() <= 100 + 14    # +14 for rounding up


def test_store_capacity_respected():
    result = optimise_orders(toy_items(), store_capacity={"CA_1": 20})
    assert result["order_qty"].sum() <= 20 + 2                              # +2 for rounding up


def test_without_limits_order_matches_newsvendor_rule():
    """s/(s+h) = 0.93 is above the 0.90 point of the scenarios, so the optimal order
    should sit at the 95th-percentile demand (15)."""
    result = optimise_orders(toy_items(n=1))
    assert abs(result["order_qty"].iloc[0] - 15) <= 1


def test_expensive_shortage_items_are_prioritised_under_tight_capacity():
    items = toy_items(n=2)
    items.loc[1, "s"] = 20.0          # item1 is much costlier to run out of
    result = optimise_orders(items, store_capacity={"CA_1": 12})
    assert result.loc[1, "order_qty"] >= result.loc[0, "order_qty"]