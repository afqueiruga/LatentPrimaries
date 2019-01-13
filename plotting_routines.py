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


def list_files(fpattern):
    files = glob.glob(fpattern)
    grab_digit = lambda f : int(re.search("([0-9]*)\.[a-zA-Z]*$",f).groups()[-1])
    files.sort(key=lambda f: grab_digit(f) )
    return files

def make_tri_plot(x,y,z, simplices, c=None, offset=(0,0), name='', **kwargs):
    return go.Mesh3d(x=x+1.5*offset[0],y=y,z=z+1.5*offset[1],
                     i=simplices[:,0],j=simplices[:,1],k=simplices[:,2],
                     intensity=c/np.max(c),
                     name=name,showscale = True,**kwargs)

def plot_ptrho(D,simplices,colorby=-1,offset=[], name='',**kwargs):
    return make_tri_plot(D[:,0],D[:,1],D[:,2], simplices, c=D[:,colorby],
                        offset=offset,name=name,**kwargs)

def plot_qqrho(D,simplices,colorby=-1,offset=[], name='',**kwargs):
    return make_tri_plot(D[:,4],D[:,5],D[:,2], simplices, c=D[:,colorby],
                        offset=offset,name=name,**kwargs)

def read_networks(training_dir):
    archs = os.listdir(training_dir)
    archs.sort()
    
    surfaces = {}
    for arch in archs:
        files = list_files(training_dir+'/'+arch+'/surf_*.csv')
        try:
            dat = np.loadtxt(files[-1],delimiter=",",skiprows=1)
        except:
            dat = np.zeros((1,3))
        surfaces[arch] = ( dat, Delaunay(dat[:,4:6]).simplices )
    return surfaces

def plot_networks(surfaces,prefix=''):
    offs = range(int(np.ceil(len(surfaces)**0.5)))
    offsets = list(itertools.product(offs,offs))
    
    ptrhos = [ plot_ptrho(d,simp,offset=o,name=n)
               for (n,(d,simp)),o in zip(surfaces.items(),offsets) ]
    fig = go.Figure(data=ptrhos)
    py.plot(fig,filename=prefix+'networks.html')
    
    qqrhos = [ plot_qqrho(d,simp,offset=o,name=n) 
               for (n,(d,simp)),o in zip(surfaces.items(),offsets) ]
    fig = go.Figure(data=ptrhos)
    py.plot(fig,filename=prefix+'networks_q.html')
    
    
def plot_simulations(database,eos_name,prefix=''):
    """Plot all of the test simulations in a grid"""
    sdb = SimDataDB(database)
    problems = sdb.Query('select distinct problem from {0}'.format(eos_name))
    networks = sdb.Query('select distinct network from {0}'.format(eos_name))
    print(problems)
    showleg = defaultdict(lambda : True)
    colorring = itertools.cycle(['black','orange','purple','red','blue','green'])
    colorkey = defaultdict(lambda : colorring.next())
    numproblems=len(problems)
    gridx = gridy = int(numproblems**0.5)
    positions = list(itertools.product(range(1,gridx+1),range(1,gridy+1)))

    subfig = tools.make_subplots(rows=4*gridx,cols=gridy,
                             shared_xaxes=False,shared_yaxes=True)
    for p,pos in zip(problems,positions):
        res = sdb.Query(
            'select network,series from {eos_name} where problem="{0}"'.
            format(p[0],eos_name=eos_name))
        legends=['T','p','rho','h']
        print p[0]
        for i,name in enumerate(legends):
            for n,t in res:
                trace = go.Scatter(x=t[:,0],y=t[:,i+3],name=n,legendgroup=n,
                                   mode='lines',
                                   line=dict(color=colorkey[n]),
                                   showlegend=showleg[n])
                showleg[n]=False
                subfig.append_trace(trace,4*(pos[0]-1)+i+1,pos[1])
    py.plot(subfig, filename=prefix+'simulation_tests.html')
    
if __name__=='__main__':
    hub = "/Users/afq/Google Drive/networks/"
    eoses = [
#         "water_slgc_logp_64",
        "water_lg",
    ]
    report_dir = hub+"report/"
    try:
        os.mkdir(report_dir)
    except OSError:
        pass
    for eos in eoses:
        surfs = read_networks(hub+'training_'+eos)
        plot_networks(surfs,prefix=report_dir+eos+'_')
        plot_simulations(hub+'test_databases/'+eos+'_testing.db',
                         eos,prefix=report_dir+eos+'_')