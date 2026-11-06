import pandas as pd
from prophet import Prophet
from src.config import HORIZON


def _prophet_one_series(ts):
    """Fit Prophet on one daily series (columns: ds, y) and forecast 28 days."""
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                    daily_seasonality=False)
    model.add_country_holidays(country_name="US")
    model.fit(ts)
    future = model.make_future_dataframe(periods=HORIZON)
    forecast = model.predict(future).tail(HORIZON)
    return forecast.set_index("ds")["yhat"].clip(lower=0)


def prophet_forecasts(history):
    """Prophet forecast for the state total and for each store total.
    Returns a dict like {"CA": series, "CA_1": series, ...}."""
    results = {}

    state = history.groupby("date")["sales"].sum().reset_index()
    state.columns = ["ds", "y"]
    results["CA"] = _prophet_one_series(state)

    for store, group in history.groupby("store_id"):
        ts = group.groupby("date")["sales"].sum().reset_index()
        ts.columns = ["ds", "y"]
        results[store] = _prophet_one_series(ts)
    return results