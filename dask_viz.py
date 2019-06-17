#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
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

multi_dropdown = html.Div([html.Div([dcc.Dropdown(id='value-selected', 
                                     options=[{'label':k,'value':k} for k in all_archs],
                                     value=all_archs[0:2], multi=True)],
                       style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"})],
             className="row")


table = dash_table.DataTable(
    id = "select-table",
    columns = [{"name": "Name","id":"id"}],
    data = [{"name":k,"id":k} for k in all_archs],
    row_selectable = "multi",
    sorting=True,
    sorting_type="multi",
    selected_rows=[0,1,2,3])

layout = html.Div([
    dcc.Graph(id="3d-graph"),
    table,
], className="two columns")


@app.callback(
    Output("3d-graph", "figure"),
    [Input("select-table", "selected_row_ids")])
def update_graph(selected):
    if selected is None: selected = []
    print(selected)
    ctx = dash.callback_context
    figure = plot_networks({k:surfs[k] for k in selected})
    return figure
    
    


app.layout = layout

if __name__ == '__main__':
    app.run_server(debug=True)