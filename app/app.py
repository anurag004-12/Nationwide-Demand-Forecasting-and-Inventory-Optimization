from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output

OUT = Path(__file__).resolve().parents[1] / "outputs"
forecasts = pd.read_csv(OUT / "forecasts.csv", parse_dates=["date"])
orders = pd.read_csv(OUT / "orders.csv")
policies = pd.read_csv(OUT / "policies.csv")
metrics = pd.read_csv(OUT / "metrics.csv")
extras = pd.read_csv(OUT / "extras.csv").set_index("item")["value"]


def table(df, **kwargs):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_cell={"textAlign": "left", "padding": "6px"},
        style_header={"fontWeight": "bold"},
        **kwargs,
    )


def card(title, value):
    return html.Div([html.Div(title, style={"fontSize": 12, "color": "#666"}),
                     html.Div(value, style={"fontSize": 22, "fontWeight": "bold"})],
                    style={"padding": "10px 16px", "border": "1px solid #ddd", "borderRadius": 8})


best = metrics[metrics["level"] == "item-store"].sort_values("WAPE").iloc[0]

app = Dash(__name__)
server = app.server            # for deployment: gunicorn app.app:server

app.layout = html.Div(style={"maxWidth": "1050px", "margin": "auto", "fontFamily": "sans-serif"}, children=[
    html.H2("Demand Forecasting & Inventory Optimization"),
    html.P("M5 Walmart, California FOODS_3, top 50 items. 28-day holdout forecast."),

    html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "16px"}, children=[
        card("Best item-level WAPE", f"{best['WAPE']:.3f} ({best['model']})"),
        card("90% interval coverage", f"{extras['coverage_90']:.1%}"),
        card("Item-state gap before", f"{extras['gap_before']:.1f} units/day"),
        card("Item-state gap after", f"{extras['gap_after']:.2f} units/day"),
    ]),

    html.H4("Forecast with 90% prediction interval"),
    dcc.Dropdown(options=sorted(forecasts["id"].unique()),
                 value=forecasts["id"].iloc[0], id="series", clearable=False),
    dcc.Graph(id="chart"),

    html.H4("Model accuracy (WAPE and RMSE; lower is better)"),
    table(metrics.round(4)),

    html.H4("Ordering policies compared against the real first week"),
    table(policies),

    html.H4("Recommended order quantities (next 7 days)"),
    dcc.Dropdown(options=["All"] + sorted(orders["store_id"].unique()),
                 value="All", id="store", clearable=False, style={"width": "200px"}),
    html.Div(id="orders_table"),
])


@app.callback(Output("chart", "figure"), Input("series", "value"))
def update_chart(series_id):
    d = forecasts[forecasts["id"] == series_id]
    fig = go.Figure()
    fig.add_scatter(x=d["date"], y=d["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=d["date"], y=d["lo"], fill="tonexty", line=dict(width=0),
                    fillcolor="rgba(0,120,255,0.2)", name="90% interval")
    fig.add_scatter(x=d["date"], y=d["forecast"], name="Forecast")
    fig.add_scatter(x=d["date"], y=d["actual"], name="Actual", mode="lines+markers")
    fig.update_layout(title=series_id, xaxis_title="Date", yaxis_title="Units sold")
    return fig


@app.callback(Output("orders_table", "children"), Input("store", "value"))
def update_orders(store):
    d = orders if store == "All" else orders[orders["store_id"] == store]
    d = d[["id", "price", "d_q50", "d_q95", "order_qty"]].round(2)
    return table(d, page_size=10, sort_action="native")


if __name__ == "__main__":
    app.run(debug=False)