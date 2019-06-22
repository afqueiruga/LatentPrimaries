#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table

external_stylesheets = ['https://codepen.io/anon/pen/mardKv.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})
# theme =  {
#     'dark': True,
#     'detail': '#007439',
#     'primary': '#00EA64',
#     'secondary': '#6E6E6E',
# }

server = app.server
app.config.suppress_callback_exceptions = True
from dash.dependencies import Input, Output, State

import plotting_routines as ls_plot
import grading_routines as ls_grade

if 'DYNO' in os.environ:
    app_name = os.environ['DASH_APP_NAME']
else:
    app_name = 'dash-text-annotationsplot'

# Configure the eos hub
hub = "/Users/afq/Google Drive/networks/"
# TODO: read this from list dir
eos_dirs = glob.glob(hub+'training_*')

eoses = [ k[(len(hub)+len('training_')):] for k in eos_dirs ]
#         "water_slgc",
#         "water_lg",
#         "water_linear",
# ]
def get_it_all(eos):
    "Load the data for a particular EOS into server memory"
    directory = hub+'training_'+eos
    # Surfaces
    surfs = ls_plot.read_networks(directory)
    surfs.pop('.DS_Store',None) # lol
    # Training and results 
    table = ls_grade.prep_table(eos,hub)
    all_archs = os.listdir(directory)
    return all_archs, table, surfs

all_archs, archs_table, surfs = {}, {}, {}
for eos in eoses:
    all_archs[eos], archs_table[eos], surfs[eos] = get_it_all(eos)
    for row in archs_table[eos]:
        row["id"]=row["name"]
    print(archs_table[eos])
##
# trash
#

# multi_dropdown = html.Div([html.Div([dcc.Dropdown(id='value-selected', 
#                                      options=[{'label':k,'value':k} for k in all_archs],
#                                      value=all_archs[0:2], multi=True)],
#                        style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"})],
#              className="row")

button = html.Button('Select All', id='my-button')
@app.callback(
    [Output('select-table', "selected_rows"),],
    [Input('my-button', 'n_clicks'),],
    [State('select-table', "derived_virtual_data"),]
)
def select_all(n_clicks, selected_rows):
    if selected_rows is None:
        return [[]]
    else:
        print(selected_rows)
        if len(selected_rows)==0:
            return [[i for i in range(len(selected_rows))]]
        else:
            return [[]]
    
# dcc.Slider(
#     min=-5,
#     max=10,
#     step=0.5,
#     value=-3,
# )

#
# end trash
##

eos_dropdown = dcc.Dropdown(id='eos-selected',
                       options=[{'label':k,'value':k} for k in eoses],
                       value=eoses[1])

# Prep the EOS selector table
columns = ls_grade.table_column_names.copy()
# columns.append("id")

table = dash_table.DataTable(
    id = "select-table",
    columns = [{"name":k,"id":k} for k in columns],
    #data = None, #filled by callback
    row_selectable = "multi",
    sorting=True,
    sorting_type="multi",
    selected_rows=[0,1,2,3])


# The page structure
layout = html.Div([
    html.Div([
        html.Div([dcc.Markdown("# EOS:")],className="two columns"),
        html.Div([eos_dropdown],className="ten columns"),
    ],className="row"),
    dcc.Graph(id="3d-graph"),
    button,
    table,
], className="row")

# Callbacks
# EOS selector refreshes the table
@app.callback(
Output("select-table","data"),[Input("eos-selected",'value')])
def update_table(eos):
    print("Selected ",eos)
    return archs_table[eos]

@app.callback(
    Output("3d-graph", "figure"),
    [Input("eos-selected",'value'),Input("select-table", "selected_row_ids")])
def update_graph(eos,selected):
    if selected is None: 
        selected = []
    print("this callback", eos, selected)
    ctx = dash.callback_context
    surfs_to_plot = {k:surfs[eos][k] for k in selected if k in surfs[eos].keys()}
    # print(surfs_to_plot)
    figure = ls_plot.plot_networks(surfs_to_plot)
    return figure
    
    


app.layout = layout
if __name__ == '__main__':
    app.run_server(debug=True)