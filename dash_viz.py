#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

from eoshub import EOSHub
import plotting_routines as ls_plot
import grading_routines as ls_grade
import test_cfg

loaded = EOSHub

#
# The components
#
# Prep the EOS selector table
columns = ls_grade.table_column_names.copy()
eos_dropdown = dcc.Dropdown(id='eos-selected',
                       options=[{'label':k,'value':k} for k in loaded.eoses],
                       value=loaded.eoses[1])

# Select graph type
select_button = html.Button('Select All', id='my-button')
graph_radio = dcc.RadioItems(
    id='graph-radio',
    options=[
        {'label': '3D rho', 'value': 'rho'},
        {'label': '3D rho_h', 'value': 'rho*h'},
        {'label': 'Traning Loss', 'value': 'training'},
        {'label': 'simulations', 'value': 'simulations'},
    ],
    value='rho',
    labelStyle={'display': 'inline-block'}
)
# Select eos problem
problem_dropdown = dcc.Dropdown(id='problem-selected',
                                options=[{'label':k,'value':k} for k in test_cfg.all_test_problems])

table = dash_table.DataTable(
    id = "select-table",
    columns = [{"name":k,"id":k} for k in columns],
    #data = None, #filled by callback
    row_selectable = "multi",
    sort_action="native",
    filter_action="native",
    #sorting=True,
    #sorting_type="multi",
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
    ROW([COL(select_button,"two"),COL([graph_radio],"two"), COL([problem_dropdown],"eight") ]),
    table,
])



#
# Callbacks
#
def gen_graph_viewport(eos,selected,z='rho'):
    surfs_to_plot = {k:loaded[eos].surfs[k] for k in selected if k in loaded[eos].surfs.keys()}
    # print(surfs_to_plot)
    figure = ls_plot.plot_networks(surfs_to_plot,z=z)
    figure.layout.template = "plotly_dark"
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
     Input("graph-radio","value"),
     ])
def update_graph(eos,selected,radio):
    if selected is None: 
        selected = []
    print("this callback", eos, selected)
    ctx = dash.callback_context
    if radio in ['rho','rho*h']:
        figure = gen_graph_viewport(eos,selected,z=radio)
    elif radio == 'training':
        lines_to_plot = {k:loaded[eos].train_scores[k] for k in selected 
                         if k in loaded[eos].train_scores.keys() }
        figure = ls_plot.make_training_plot(lines_to_plot)
    elif radio == 'simulation':
        sdb = SimDataDB()
        
        arch_filter = [ k for k in selected if k in loaded[eos].train_scores.keys() ]
        figure = ls_plot.make_training_plot(lines_to_plot)
        
    else:
        problem = "Liquid_Drain"
        print(selected)
        try:
            figure = ls_plot.plotyly_query_simulations("Liquid_Drain",eos, 
                                                   arch_filter=selected)
        except:
            print(f"No entry found for {eos} solving {problem}")
            figure = None
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