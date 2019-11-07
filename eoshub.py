from collections import namedtuple
import os, glob

from directory_parsing import hubs

import grading_routines as ls_grade
import plotting_routines as ls_plot
import test_cfg
from latent_sim import LatentSim

class LazyLoad():
    "Lazily load the state of all of the eoses"
    Entry = namedtuple('Entry', ['archs', 'table','surfs','train_scores'])
    def __init__(self,hubs):
        self.eoses = []
        self.cache = {} # Cache processing results
        self.hubs = {} # Mapping from eos to where to find it
        self.sims = {} # Cache of LatentSim objects
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
        
    def eos_loadout(self, eos, network=None):
        """
        Parse the database and get the arguments that construct
        the simulator.
        """
        if network is None:
            network = self[eos].archs[0]
        for dataset_match in test_cfg.eos_test_cfg:
            if dataset_match in eos:
                scale_file = test_cfg.eos_test_cfg[dataset_match]['scale_file']
                logp = test_cfg.eos_test_cfg[dataset_match]['logp']
                break
        args = (self._eos_dir(eos)+'/'+network,scale_file,logp)
        return args
    
    def LatentSim(self, eos,network=None):
        """
        Create a latent sim object, or fetch-and-wipe one that's
        stored in memory
        """
        if network is None:
            network = self[eos].archs[0]
        try:
            ls = self.sims[(eos,network)]
        except KeyError:
            args = eos_loadout(eos, network)
            ls = LatentSim(*args)
            self.sims[(eos,network)] = ls
        return ls
    
EOSHub = LazyLoad(hubs)