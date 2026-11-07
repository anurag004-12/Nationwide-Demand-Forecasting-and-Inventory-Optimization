import numpy as np
from ortools.linear_solver import pywraplp

SCENARIO_COLS = ["d_q05", "d_q25", "d_q50", "d_q75", "d_q95"]
WEIGHTS = [0.10, 0.20, 0.40, 0.20, 0.10]     # how likely each demand scenario is


def make_item_table(valid):
    """Total demand for the first 7 days, for each quantile, plus cost assumptions."""
    first_week = valid[valid["date"] < valid["date"].min() + np.timedelta64(7, "D")]
    items = first_week.groupby("id").agg(
        price=("sell_price", "mean"),
        d_q05=("q05", "sum"), d_q25=("q25", "sum"), d_q50=("q50", "sum"),
        d_q75=("q75", "sum"), d_q95=("q95", "sum"),
    ).reset_index()

    # Cost assumptions (write these in your README; they are inputs you can change)
    items["unit_cost"] = 0.7 * items["price"]                       # what we pay per unit
    items["h"] = 0.05 * items["unit_cost"]                          # cost of one leftover unit
    items["s"] = 1.5 * (items["price"] - items["unit_cost"])        # cost of one missed sale
    return items


def optimise_orders(items, budget):
    """Choose order quantities that minimise expected leftover + shortage cost,
    while total purchase cost stays within the budget."""
    solver = pywraplp.Solver.CreateSolver("GLOP")
    inf = solver.infinity()

    order = [solver.NumVar(0, inf, f"order_{i}") for i in range(len(items))]
    objective = solver.Objective()

    for i, row in enumerate(items.itertuples()):
        for col, weight in zip(SCENARIO_COLS, WEIGHTS):
            demand = getattr(row, col)
            leftover = solver.NumVar(0, inf, "")      # units we ordered but did not sell
            missed = solver.NumVar(0, inf, "")        # units customers wanted but we lacked
            solver.Add(leftover >= order[i] - demand)
            solver.Add(missed >= demand - order[i])
            objective.SetCoefficient(leftover, weight * row.h)
            objective.SetCoefficient(missed, weight * row.s)

    # Budget constraint
    total_cost = sum(row.unit_cost * order[i] for i, row in enumerate(items.itertuples()))
    solver.Add(total_cost <= budget)

    objective.SetMinimization()
    status = solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL, "Solver did not find an optimal solution"

    result = items.copy()
    result["order_qty"] = np.ceil([v.solution_value() for v in order])
    return result