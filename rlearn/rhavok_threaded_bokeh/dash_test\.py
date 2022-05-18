#!/usr/bin/env python3

import random
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output



# Read data from a csv
z_data = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')


import numpy as np 
mu, sigma = 0, 50 
# creating a noise with the same dimension as the dataset (2,2) 
noise = np.random.normal(mu, sigma, z_data.shape)
s1 = go.Surface(z=((z_data+noise)/(z_data+noise).max().max()).values)
noise = np.random.normal(mu, sigma, z_data.shape)
s2 = go.Surface(z=((z_data+noise)/(z_data+noise).max().max()).values)
noise = np.random.normal(mu, sigma, z_data.shape)
s3 = go.Surface(z=((z_data+noise)/(z_data+noise).max().max()).values)
noise = np.random.normal(mu, sigma, z_data.shape)
s4 = go.Surface(z=((z_data+noise)/(z_data+noise).max().max()).values)
noise = np.random.normal(mu, sigma, z_data.shape)
s5 = go.Surface(z=((z_data+noise)/(z_data+noise).max().max()).values)

frames = [{"data":[s1]},{"data":[s2]},{"data":[s3]},{"data":[s4]},{"data":[s5]}]

fig = go.Figure(data=[s1],layout=go.Layout(
        xaxis=dict(range=[0, 16], autorange=False),
        yaxis=dict(range=[0, 13], autorange=False),
        hovermode="closest",
        title="Start Title"
    ))


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc.Graph(id='live-graph'),
        dcc.Interval(
            id='graph-update',
            interval=100
        ),
    ]
)

@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_graph_scatter(input_data):
    if input_data is None:
        return dash.no_update
    return frames[input_data % 5]


if __name__ == '__main__':
    app.run_server(debug=True)
