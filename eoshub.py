from collections import namedtuple
import os, glob

import grading_routines as ls_grade

# Configure the eos hub
hubs = [
    "/Users/afq/Google Drive/networks/",
    "/Users/afq/Documents/Research/LBNL/eoshub/eoshub/networks/",
    "/Users/afq/Research/eoshub/networks/",   
]

class LazyLoad():
    "Lazily load the state of all of the eoses"
    Entry = namedtuple('Entry', ['archs', 'table','surfs','train_scores'])
    def __init__(self,hubs):
        self.eoses = []
        self.cache = {} # Cache processing results
        self.hubs = {} # Mapping from eos to where to find it
        for hub in hubs:
            eos_dirs = glob.glob(hub+'/training_*')
            eoses = [ k[(len(hub)+len('training_')):] 
                      for k in eos_dirs ]
            self.eoses.extend(eoses)
            for e in eoses:
                self.hubs[e] = hub
        print(self.hubs)
        
    def _eos_dir(self,eos):
        return self.hubs[eos]+'/training_'+eos
    
    def get_it_all(self,eos):
        "Load the data for a particular EOS into server memory"
        directory = self._eos_dir(eos)
        # Surfaces
        surfs = ls_plot.read_networks(directory)
        surfs.pop('.DS_Store',None) # lol
        # Training and results 
        
        table, train_scores = ls_grade.prep_table(eos,self.hubs[eos])
        all_archs = os.listdir(directory)
        # TODO: Get the simulation results
        # TODO: Run the simulations in batch
        for row in table:
            row["id"]=row["name"]
        return self.Entry(all_archs, table, surfs, train_scores)
    
    def __getitem__(self, key):
        try:
            return self.cache[key]
        except KeyError:
            rez = self.get_it_all(key)
            self.cache[key] = rez
            return rez
        
EOSHub = LazyLoad(hubs)