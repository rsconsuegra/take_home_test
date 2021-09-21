import pickle

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State


with open("bayesian_ridge.pickle", "rb") as handle:
    brc = pickle.load(handle)

weights = pd.read_pickle("attribute_weight")

fig = px.bar(
    weights,
    orientation="h",
    width=800,
    height=800,
    title="Model Coefficients impact on attributes",
)

seller_loyalty = [
    "bronze",
    "free",
    "gold",
    "gold_premium",
    "gold_pro",
    "gold_special",
    "silver",
]


buying_mode = ["auction", "buy_it_now", "classified"]

shipping_mode = ["custom", "me1", "me2", "not_specified"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label(f"Choose a Price between 0-267201"),
                dbc.Input(id="price", type="number", min=0, max=267201),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Seller Loyalty"),
                dcc.Dropdown(
                    id="seller_loyalty",
                    options=[{"label": col, "value": col} for col in seller_loyalty],
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Buying Mode"),
                dcc.Dropdown(
                    id="buying_mode",
                    options=[{"label": col, "value": col} for col in buying_mode],
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Shipping Mode"),
                dcc.Dropdown(
                    id="shipping_mode",
                    options=[{"label": col, "value": col} for col in shipping_mode],
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Admits Pickup?"),
                daq.BooleanSwitch(id="pickup-switch", on=True),
            ]
        ),
        dbc.FormGroup(
            [dbc.Label("Free Shipping?"), daq.BooleanSwitch(id="free-switch", on=True),]
        ),
        dbc.FormGroup(
            [dbc.Label("Is New?"), daq.BooleanSwitch(id="new-switch", on=True),]
        ),
        dbc.FormGroup(
            [
                dbc.Label(f"Choose a Initial quantity between 0-1000"),
                dcc.Input(
                    id="init_quant", type="number", min=0, max=1000, required=True
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label(f"Choose a number of pictures in publication between 0-36"),
                dbc.Input(id="num_of_pic", type="number", min=0, max=36),
            ]
        ),
        dbc.Alert(
            "Please, insert a valid value!",
            id="alert-auto",
            is_open=True,
            duration=1000,
            color="danger",
        ),
        dbc.Button("Do Prediction!", color="primary", id="button-predict"),
        html.Div(id="boolean-switch-output"),
    ],
    body=True,
)

app.layout = dbc.Container(
    [
        html.H1("MeLI"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(dcc.Graph(id="cluster-graph", figure=fig), md=8),
            ],
            align="center",
        ),
    ],
    fluid=True,
)


@app.callback(
    [Output("boolean-switch-output", "children"), Output("alert-auto", "is_open")],
    [Input("button-predict", "n_clicks")],
    [
        State("price", "value"),
        State("init_quant", "value"),
        State("num_of_pic", "value"),
        State("seller_loyalty", "value"),
        State("buying_mode", "value"),
        State("shipping_mode", "value"),
        State("pickup-switch", "on"),
        State("free-switch", "on"),
        State("new-switch", "on"),
    ],
)
def compute_prediction(
    n_clicks,
    price,
    init_quant,
    num_of_pic,
    seller_loyalty_value,
    buying_mode_value,
    shipping_mode_value,
    pickup,
    free,
    new,
):
    if [x for x in (price, init_quant, num_of_pic) if x is None]:
        # PreventUpdate prevents ALL outputs updating
        return "", True

    loyalty_encoding = [0] * len(seller_loyalty)
    loyalty_encoding[seller_loyalty.index(seller_loyalty_value)] = 1

    buying_mode_encoding = [0] * len(buying_mode)
    buying_mode_encoding[buying_mode.index(buying_mode_value)] = 1

    shipping_mode_encoding = [0] * len(shipping_mode)
    shipping_mode_encoding[shipping_mode.index(shipping_mode_value)] = 1

    pickup_encoding = []
    if pickup:
        pickup_encoding = [0, 1]
    else:
        pickup_encoding = [1, 0]

    free_encoding = []
    if free:
        free_encoding = [0, 1]
    else:
        free_encoding = [1, 0]

    status_encoding = [1, 0, 0]

    new_encoding = []
    if new:
        new_encoding = [0, 1]
    else:
        new_encoding = [1, 0]

    sample = (
        np.concatenate(
            [
                [price, init_quant, num_of_pic],
                loyalty_encoding,
                buying_mode_encoding,
                shipping_mode_encoding,
                pickup_encoding,
                free_encoding,
                status_encoding,
                new_encoding,
            ]
        )
        .ravel()
        .reshape(1, -1)
    )

    prediction = brc.predict(sample)
    message = ""
    if prediction > 0.2:
        message = "You will probably have, at least, one sell with this publication"
    else:
        message = "You will probably not sell with this publication"

    return message, False


if __name__ == "__main__":
    app.run_server(debug=False)
