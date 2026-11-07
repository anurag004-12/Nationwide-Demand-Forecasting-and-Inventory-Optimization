import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

SCENARIO_COLS = ["d_q05", "d_q25", "d_q50", "d_q75", "d_q95"]
WEIGHTS = [0.10, 0.20, 0.40, 0.20, 0.10]     # how likely each demand scenario is
ORDER_DAYS = 7                                # we order for the next 7 days


def make_item_table(valid):
    """Total demand for the first 7 days, for each quantile, plus cost assumptions.
    One row per item-store series."""
    start = valid["date"].min()
    first_week = valid[valid["date"] < start + pd.Timedelta(days=ORDER_DAYS)]

    items = first_week.groupby("id").agg(
        store_id=("store_id", "first"),
        price=("sell_price", "mean"),
        d_q05=("q05", "sum"), d_q25=("q25", "sum"), d_q50=("q50", "sum"),
        d_q75=("q75", "sum"), d_q95=("q95", "sum"),
    ).reset_index()

    # Cost assumptions (write these in your README; they are inputs you can change)
    items["unit_cost"] = 0.7 * items["price"]                       # what we pay per unit
    items["h"] = 0.05 * items["unit_cost"]                          # cost of one leftover unit
    items["s"] = 1.5 * (items["price"] - items["unit_cost"])        # cost of one missed sale
    return items


def shrink_spread(items, factor):
    """Pull the quantile scenarios closer to the median.
    Summing 7 daily quantiles overstates the weekly spread, so factor < 1 shrinks it.
    factor = 1.0 leaves the table unchanged."""
    out = items.copy()
    for col in ["d_q05", "d_q25", "d_q75", "d_q95"]:
        out[col] = out["d_q50"] + factor * (out[col] - out["d_q50"])
    return out


def optimise_orders(items, budget=None, store_capacity=None):
    """Choose order quantities that minimise expected leftover + shortage cost.

    budget         : max total purchase cost (money), or None for no limit
    store_capacity : dict like {"CA_1": 900, ...} = max total units per store, or None
    """
    solver = pywraplp.Solver.CreateSolver("GLOP")
    inf = solver.infinity()

    rows = list(items.itertuples())
    order = [solver.NumVar(0, inf, f"order_{i}") for i in range(len(rows))]
    objective = solver.Objective()

    for i, row in enumerate(rows):
        for col, weight in zip(SCENARIO_COLS, WEIGHTS):
            demand = getattr(row, col)
            leftover = solver.NumVar(0, inf, "")      # units we ordered but did not sell
            missed = solver.NumVar(0, inf, "")        # units customers wanted but we lacked
            solver.Add(leftover >= order[i] - demand)
            solver.Add(missed >= demand - order[i])
            objective.SetCoefficient(leftover, weight * row.h)
            objective.SetCoefficient(missed, weight * row.s)

    # Constraint 1: total money spent
    if budget is not None:
        total_cost = sum(row.unit_cost * order[i] for i, row in enumerate(rows))
        solver.Add(total_cost <= budget)

    # Constraint 2: shelf space per store, in units
    if store_capacity:
        stores = items["store_id"].tolist()
        for store, cap in store_capacity.items():
            idx = [i for i, s in enumerate(stores) if s == store]
            solver.Add(sum(order[i] for i in idx) <= cap)

    objective.SetMinimization()
    status = solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL, "Solver did not find an optimal solution"

    result = items.copy()
    result["order_qty"] = np.ceil([v.solution_value() for v in order])
    return result


def weekly_actuals(valid):
    """Real units sold in the first 7 days of the holdout period, per series."""
    start = valid["date"].min()
    week = valid[valid["date"] < start + pd.Timedelta(days=ORDER_DAYS)]
    return week.groupby("id")["sales"].sum().rename("actual")


def evaluate_policy(orders, actual, name):
    """Score an order plan against what actually sold.
    `orders` needs columns: id, order_qty, s, h."""
    d = orders.merge(actual, on="id")
    short = (d["actual"] - d["order_qty"]).clip(lower=0)     # missed sales
    left = (d["order_qty"] - d["actual"]).clip(lower=0)      # unsold units
    return {
        "policy": name,
        "fill_rate": round(1 - short.sum() / d["actual"].sum(), 3),
        "leftover_units": int(left.sum()),
        "total_cost": round(((short * d["s"]) + (left * d["h"])).sum(), 1),
    }