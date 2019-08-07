import os, glob, re
import itertools
from collections import defaultdict

import numpy as np
from scipy.spatial import Delaunay

from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
# py.init_notebook_mode(connected=True)

from SimDataDB import SimDataDB

color_ring = itertools.cycle(['black','orange','purple','red','blue','green'])
colorkey = defaultdict(lambda : next(color_ring)) # It should be global, no?
d_off = 2.0

from directory_parsing import list_files

#
# Plotting of all of the scores
#
def make_training_plot(train_scores):
    layout = dict(
        yaxis=dict(title='Test loss',type='log'),
        xaxis=dict(title='Step number'))
    return go.Figure(data=[ go.Scatter(x=v[:,0],y=v[:,1],name=k)
                            for k,v in train_scores.items() ], layout=layout)

#
# 3D Plotting of the learned surfaces
#
def make_tri_plot(x,y,z, simplices, c=None, offset=(0,0), name='', **kwargs):
    return go.Mesh3d(x=x+d_off*offset[0],y=y,z=z+d_off*offset[1],
                     i=simplices[:,0],j=simplices[:,1],k=simplices[:,2],
                     intensity=(1.0*c)/np.max(1.0*c),
                     name=name,showscale = True,
                     colorscale='Jet',
                     **kwargs)

def plot_ptrho(D,simplices,colorby=-1,offset=(0,0), name='', **kwargs):
    return make_tri_plot(D[:,0],D[:,1],D[:,2], simplices, c=D[:,colorby],
                        offset=offset,name=name,**kwargs)

def plot_pth(D,simplices,colorby=-1,offset=(0,0), name='', **kwargs):
    return make_tri_plot(D[:,0],D[:,1],D[:,3], simplices, c=D[:,colorby],
                        offset=offset,name=name,**kwargs)

def plot_qqrho(D,simplices,colorby=-1,offset=(0,0), name='', **kwargs):
    return make_tri_plot(D[:,4],D[:,5],D[:,2], simplices, c=D[:,colorby],
                        offset=offset,name=name,**kwargs)

def plot_qq_Tprhoh(D,simplices,colorby=-1,offset=(0,0), name='', **kwargs):
    plts = []
#     subfig = tools.make_subplots(rows=2,cols=2,
#         subplot_titles=('T','p','rho','h'),
#         specs=[
#         [{'is_3d': True}, {'is_3d': True}],
#         [{'is_3d': True}, {'is_3d': True}]
#     ])

    for i in range(4):
        offset_i = (offset[0] + 2*(i%2), offset[1] + 1.25*(i/2))
        tp = make_tri_plot(D[:,4],D[:,5],D[:,i], simplices, c=D[:,colorby],
                        offset=offset_i,name=name,**kwargs)
        plts.append(tp)
    return plts

def read_all_surfaces(arch_dir):
    files = list_files(arch_dir+'/surf_*.csv')
    surfs = []
    for f in files:
        dat = np.loadtxt(f,delimiter=",",skiprows=1)
        surfs.append( ( dat, Delaunay(dat[:,4:6]).simplices ) )
    return surfs

def make_animation(surfs):
    return go.Figure(data=[plot_ptrho(surfs[0][0],surfs[0][1])],
                     layout=dict(
                         updatemenus= [{'type': 'buttons',
                           'buttons': [{'label': 'Play',
                                        'method': 'animate',
                                        'args': [None]}]}]),
              frames=[{"data":[plot_ptrho(d,s)]} for d,s in surfs[1::20]])

def read_networks(training_dir):
    archs = os.listdir(training_dir)
    archs.sort()
    
    surfaces = {}
    for arch in archs:
        files = list_files(training_dir+'/'+arch+'/surf_*.csv')
        try:
            dat = np.loadtxt(files[-1],delimiter=",",skiprows=1)
            surfaces[arch] = ( dat, Delaunay(dat[:,4:6]).simplices )
        except:
            print("No surface for ",arch)
            #dat = np.zeros((3,7))
            #surfaces[arch] = ( dat, [] )
    return surfaces

def generate_trisurf_plots(surfaces):
    return [ (n,go.Figure(data=[plot_ptrho(d,simp,name=n)]))
             for n,(d,simp) in surfaces.items() ]

def plot_networks(surfaces,aspectratio=1, z='rho'):
    offs = range(int(np.ceil(len(surfaces)**0.5)))
    offsets = list(itertools.product(offs,offs))
    if z=='rho':
        PLT = plot_ptrho
    else:
        PLT = plot_pth
    ptrhos = [ PLT(d,simp,offset=(o[1],o[0]),name=n)
               for (n,(d,simp)),o in zip(surfaces.items(),offsets) ]
    annotes = [ dict(text=n,x=d_off*o[1],y=0,z=d_off*o[0])
                for n,o in zip(surfaces,offsets) ]
    fig = go.Figure(data=ptrhos,
                    layout=go.Layout(
                        scene=dict(annotations=annotes,
                                   aspectmode='data'),
                        margin=dict(r=5, l=5,b=5, t=5)) )
    return fig

#     qqrhos = [ plot_qqrho(d,simp,offset=o,name=n) 
#                for (n,(d,simp)),o in zip(surfaces.items(),offsets) ]
#     fig = go.Figure(data=ptrhos)
#     py.plot(fig,filename=prefix+'networks_q.html')



#
# Simulations
#
def plotly_simulation_t(sims,ref=None, showlegend=False):
    legends=['T','p','rho','rho*h']
    traces = []
    for i,name in enumerate(legends):
        if not ref is None:
            trace_ref = go.Scatter(x=ref[:,0],
                                  y=ref[:,i+1],
                                  name='ref',legendgroup='ref',
                                  showlegend= (i==0 and showlegend),
                                  line=dict(dash='dash', color=colorkey['ref']))
            traces.append((trace_ref,i+1))
        for n,time_series in sims.items():
            trace = go.Scatter(x=time_series[:,0],y=time_series[:,i+3],
                               name=n,legendgroup=n,
                               line=dict(color=colorkey[n]),
                               showlegend=(i==0 and showlegend))
            traces.append((trace,i+1))
    return traces


def plotly_simulations_Tp(sims,ref=None,showlegend=True):
    from equations_of_state.iapws_boundaries \
        import plot_boundaries_plotly
    traces_bound = plot_boundaries_plotly()
    layout=dict(yaxis=dict(title='log(p)',type='log'),
            xaxis=dict(title='T'))
    if not ref is None:
        trace_ref = go.Scatter(x=ref[:,1],
                       y=ref[:,2],
                       name='ref',legendgroup='ref',
                       showlegend=showlegend,
                       line=dict(dash='dash',color=colorkey['ref']))
    traces_sims = []
    for n,time_series in sims.items():
        trace_num = go.Scatter(x=time_series[:,3],
                       y=time_series[:,4],
                       name=n,legendgroup=n,
                       showlegend=showlegend,
                       line=dict(color=colorkey[n]))
        traces_sims.append(trace_num)
    traces = traces_bound + traces_sims + ([trace_ref] if not ref is None else [])
    #fig = go.Figure(data=trace_bound+[trace_ref,trace_num],layout=layout)
    return traces, layout


def plotly_simulations_Tp_ts(sims,ref=None,showlegend=True):
    trace_Tp, layout_Tp = plotly_simulations_Tp(sims,ref,showlegend=False)
    traces_ts = plotly_simulation_t(sims,ref,showlegend=showlegend)
    subfig = tools.make_subplots(rows=4,cols=2,
                specs=[[{'rowspan':4,'colspan':1}, {}],
                [None, {}],
                [None, {}],
                [None, {}],],
                shared_xaxes=False)
    subfig['layout']['yaxis1']['type']='log'
    subfig['layout']['yaxis1']['title']='log(p)'
    subfig['layout']['xaxis1']['title']='T (K)'
    subfig['layout']['xaxis5']['title']='t (s)'
    subfig['layout']['yaxis2']['title']='T'
    subfig['layout']['yaxis3']['title']='P'
    subfig['layout']['yaxis4']['title']='rho'
    subfig['layout']['yaxis5']['title']='rho*h'
    for trace in trace_Tp:
        subfig.append_trace(trace,1,1)
    for trace,x in traces_ts:
        subfig.append_trace(trace,x,2)
    return subfig
    
    

def plotyly_query_simulations(problem_name,eos,reference=None,result_dir='./'):
    sdb = SimDataDB(result_dir+f'/{eos}_testing.db')
    q = sdb.Query(f'select network,series from {eos} where problem=="{problem_name}"')
    fig = plotly_simulations_Tp_ts({k:v for k,v in q},
                                  ref_arr_prep)
    return fig
    
def plot_one_simulation(sdb, eos_name, problem): # DEP
    networks = sdb.Query('select distinct network from {0}'.format(eos_name))
    legends = ['T','p','rho','h']
    Nfields = len(legends)
    gridx, gridy = 1,Nfields
    showleg = defaultdict(lambda : True)
    
    subfig = tools.make_subplots(rows=gridy,cols=gridx,
                                 shared_xaxes=True,shared_yaxes=False)
    res = sdb.Query(
            'select network,series from {eos_name} where problem="{0}"'.
            format(problem,eos_name=eos_name))
    for i,name in enumerate(legends):
        for n,t in res:
            trace = go.Scatter(x=t[:,0],y=t[:,i+3],name=n,legendgroup=n,
                               mode='lines',
                               line=dict(color=colorkey[n]),
                               showlegend=showleg[n])
            showleg[n]=False
            subfig.append_trace(trace,1+i,1)
    return subfig

def plot_pT_simulation(sdb, eos_name, problem): # DEP
    networks = sdb.Query('select distinct network from {0}'.format(eos_name))
    showleg = defaultdict(lambda : True)
    res = sdb.Query(
            'select network,series from {eos_name} where problem="{0}"'.
            format(problem,eos_name=eos_name))
    traces = []
    for n,t in res:
        traces.append( 
            go.Scatter(x=t[:,3],y=t[:,4],name=n,legendgroup=n,
            mode='lines',
            line=dict(color=colorkey[n]),
            showlegend=False) )
    return traces



def make_simulation_plot_list(database,eos_name): # DEP
    sdb = SimDataDB(database)
    problems = sdb.Query('select distinct problem from {0}'.format(eos_name))
    print(problems)
    plots = [ plot_one_simulation(sdb,eos_name, p[0]) 
             for p in problems ]
    return plots


        
def plot_simulations(database,eos_name,prefix=''): # DEP
    """Plot all of the test simulations in a grid"""
    sdb = SimDataDB(database)
    problems = sdb.Query('select distinct problem from {0}'.format(eos_name))
    networks = sdb.Query('select distinct network from {0}'.format(eos_name))
    print(problems)
    showleg = defaultdict(lambda : True)
    numproblems=len(problems)
    gridx = int(numproblems**0.5)+1
    gridy = int(numproblems / gridx)+1
    positions = list(itertools.product(range(1,gridx+1),range(1,gridy+1)))

    #     titles, titles_trans = [], range(4*gridx*gridy)
    #     for prob in problems:
    #         titles.extend([prob[0]] + ['_' for _ in range(4-1) ])
    #     for i in range(4*gridx):
    #         for j in range(gridy):
    #             titles_trans[j*4*gridx+i] = titles[4*i*gridy+j]
    titles = ['' for _ in range(4*gridx*gridy)]
    for p,pos in zip(problems,positions):
        #titles[4*gridx*(pos[1]-1)+(pos[0]-1)+1] = p[0]
        titles[4*gridy*(pos[0]-1)+(pos[1]-1)] = p[0]
    subfig = tools.make_subplots(rows=4*gridx,cols=gridy,
                  shared_xaxes=False, shared_yaxes=False,
                  subplot_titles=titles)
    for p,pos in zip(problems,positions):
        res = sdb.Query(
            'select network,series from {eos_name} where problem="{0}"'.
            format(p[0],eos_name=eos_name))
        legends=['T','p','rho','h']
        #print p[0]
        for i,name in enumerate(legends):
            for n,t in res:
                trace = go.Scatter(x=t[:,0],y=t[:,i+3],name=n,legendgroup=n,
                                   mode='lines',
                                   line=dict(color=colorkey[n]),
                                   showlegend=showleg[n])
                showleg[n]=False
                subfig.append_trace(trace,4*(pos[0]-1)+i+1,pos[1])
    # from IPython import embed ; embed()
    return subfig



#
# Generate a be-all-end-all report
#
if __name__=='__main__':
    hub = "/Users/afq/Google Drive/networks/"
    eoses = [
        "water_slgc_logp_64",
        "water_lg",
        "water_linear",
    ]
    report_dir = hub+"report/"
    try:
        os.mkdir(report_dir)
    except OSError:
        pass
    for eos in eoses:
        prefix = report_dir+eos+'_'
#         surfs = read_networks(hub+'training_'+eos)
        
#         netplots = plot_networks(surfs)
#         py.plot(fig,filename=prefix+'networks.html')
        
        simplots = plot_simulations(hub+'test_databases/'+eos+'_testing.db',eos)
        py.plot(simplots, filename=prefix+'simulation_tests.html')
        
#         qqplot = plot_qq_Tprhoh(*surfs['Classifying_2,4,12,24,sigmoid (1)']) 
#         py.plot(go.Figure(qqplot), filename=prefix+'qq_plots.html')
        