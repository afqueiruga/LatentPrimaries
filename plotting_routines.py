import numpy as np
import os, glob, re
import itertools
from collections import defaultdict

from scipy.spatial import Delaunay
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
# py.init_notebook_mode(connected=True)


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
    
    
def plot_simulations(database,prefix=''):
    pass
if __name__=='__main__':
    hub = "/Users/afq/Google Drive/networks/"
    eoses = [
        "training_water_slgc_logp_64",
        "training_water_lg",
    ]
    
    for eos in eoses:
        surfs = read_networks(hub+eos)
        plot_networks(surfs,prefix=eos+'_')
    