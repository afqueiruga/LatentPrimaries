#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple
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

class LazyLoad():
    "Lazily load the state of all of the eoses"
    Entry = namedtuple('Entry', ['archs', 'table','surfs'])
    def __init__(self,hub):
        eos_dirs = glob.glob(hub+'training_*')
        self.hub = hub
        self.eoses = [ k[(len(hub)+len('training_')):] for k in eos_dirs ]
        self.cache = {}
        
    def get_it_all(self,eos):
        "Load the data for a particular EOS into server memory"
        directory = self.hub+'training_'+eos
        # Surfaces
        surfs = ls_plot.read_networks(directory)
        surfs.pop('.DS_Store',None) # lol
        # Training and results 
        table = ls_grade.prep_table(eos,self.hub)
        all_archs = os.listdir(directory)
        for row in table:
            row["id"]=row["name"]
        return self.Entry(all_archs, table, surfs)
    
    def __getitem__(self, key):
        try:
            return self.cache[key]
        except KeyError:
            rez = self.get_it_all(key)
            self.cache[key] = rez
            return rez

loaded = LazyLoad(hub)

# Prep the EOS selector table
columns = ls_grade.table_column_names.copy()
eos_dropdown = dcc.Dropdown(id='eos-selected',
                       options=[{'label':k,'value':k} for k in loaded.eoses],
                       value=loaded.eoses[1])

select_button = html.Button('Select All', id='my-button')
graph_radio = dcc.RadioItems(
    options=[
        {'label': '3D rho', 'value': 'rho'},
        {'label': '3D rho_h', 'value': 'rho_h'},
        {'label': 'simulations', 'value': 'simulations'}
    ],
    value='rho',
    labelStyle={'display': 'inline-block'}
)

table = dash_table.DataTable(
    id = "select-table",
    columns = [{"name":k,"id":k} for k in columns],
    #data = None, #filled by callback
    row_selectable = "multi",
    sorting=True,
    sorting_type="multi",
    selected_rows=[0,1,2,3])



#
# The page layout
#
# div macros
ROW = lambda l : html.Div(l,className="row")
COL = lambda l, num="one" : html.Div(l,className=num+" columns")
# The page structure
layout = ROW([
    ROW([
        COL([dcc.Markdown("# EOS:")],"two"),
        COL([eos_dropdown],"ten"),
    ]),
    dcc.Graph(id="3d-graph"),
    ROW([COL(select_button,"two"),COL([graph_radio],"ten")]),
    table,
])



#
# Callbacks
#
def gen_graph_viewport(eos,selected):
    surfs_to_plot = {k:loaded[eos].surfs[k] for k in selected if k in loaded[eos].surfs.keys()}
    # print(surfs_to_plot)
    figure = ls_plot.plot_networks(surfs_to_plot)
    return figure

# EOS selector refreshes the table
@app.callback(
Output("select-table","data"),[Input("eos-selected",'value')])
def update_table(eos):
    print("Selected ",eos)
    return loaded[eos].table

# Select All/None button
@app.callback(
    [Output('select-table', "selected_rows")],
    [Input('my-button', 'n_clicks'),],
    [State('select-table', "derived_virtual_data"),
     State('select-table', "derived_virtual_selected_rows"),]
)
def select_all(n_clicks, all_rows, selected_rows):
    if selected_rows is None:
        newrows = [[]]
    else:
        print(selected_rows)
        if len(selected_rows)==0:
            newrows = [[i for i in range(len(all_rows))]]
        else:
            newrows = [[]]
    #figure = gen_graph_viewport(eos,selected)
    return newrows

# Select individual graphs
@app.callback(
    Output("3d-graph", "figure"),
    [Input("eos-selected",'value'),
     Input("select-table", "selected_row_ids"),
     ])
def update_graph(eos,selected):
    if selected is None: 
        selected = []
    print("this callback", eos, selected)
    ctx = dash.callback_context
    figure = gen_graph_viewport(eos,selected)
    return figure
    
    
app.layout = layout
if __name__ == '__main__':
    app.run_server(debug=True)


##
# trash
#
# multi_dropdown = html.Div([html.Div([dcc.Dropdown(id='value-selected', 
#                                      options=[{'label':k,'value':k} for k in all_archs],
#                                      value=all_archs[0:2], multi=True)],
#                        style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"})],
#              className="row")
# dcc.Slider(
#     min=-5,
#     max=10,
#     step=0.5,
#     value=-3,
# )
#
# end trash
##