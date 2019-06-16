#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True
from dash.dependencies import Input, Output, State

from plotting_routines import *

if 'DYNO' in os.environ:
    app_name = os.environ['DASH_APP_NAME']
else:
    app_name = 'dash-text-annotationsplot'

# Configure the eos hub
hub = "/Users/afq/Google Drive/networks/"
eoses = [
        "water_slgc_logp_64",
        "water_lg",
        "water_linear",
]
eos = eoses[1]
# Surfs
surfs = read_networks(hub+'training_'+eos)
surfs.pop('.DS_Store',None) # lol

all_archs = list(surfs.keys())


layout = html.Div([
    html.Div([html.H1("Your 3D surfaces")], style={'textAlign': "center"}),
    html.Div([html.Div([dcc.Dropdown(id='value-selected', 
                                     options=[{'label':k,'value':k} for k in all_archs],
                                     value=all_archs[0:2], multi=True)],
                       style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"})],
             className="row"),
    dcc.Graph(id="my-graph"),
], className="six columns")


@app.callback(
    Output("my-graph", "figure"),
    [Input("value-selected", "value")])
def update_graph(selected):
    dropdown = {k:k for k in all_archs}
    ctx = dash.callback_context
    figure = plot_networks({k:surfs[k] for k in selected})
    return figure
    
    


app.layout = layout

if __name__ == '__main__':
    app.run_server(debug=True)