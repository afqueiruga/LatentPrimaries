from __future__ import print_function
import sys,inspect

class Linear_Liquid():
    t_max = 10.0
    initial = dict(p = 1.0e5, T = 20.0+273.15)
    params =  dict(k_p=1.0e-4,k_T=1.0e4)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=293.15,p_inf=1.0e5+10.0)
    
class Small_Liquid():
    t_max = 1000.0
    initial = dict(T=350,p=5.0e5)
    params =  dict(k_p=1.0e-4,k_T=1.0e4)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=450,p_inf=5.0e5)

class Small_Gas():
    t_max = 1000.0
    initial = dict(T=350,p=5.0e3)
    params =  dict(k_p=1.0e-4,k_T=1.0e4)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=450,p_inf=5.0e3)

class Hot_Gas():
    t_max = 1000.0
    initial = dict(T=450,p=5.0e6)
    params =  dict(k_p=1.0e-4,k_T=1.0e4)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=550,p_inf=5.0e6)

        
class Transition_L2G():
    t_max = 1000.0
    initial = dict(T=350,p=5.0e5)
    params =  dict(k_p=1.0e-4,k_T=1.0e4)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=350,p_inf=5.0e3)

class Cycle_sgclg():
    t_max = 100.0
    initial = dict(T=250,p=5.0e3)
    params =  dict(k_p=1.0e-4,k_T=1.0e3)
    @staticmethod
    def schedule(sim,t):
        if t<1000.0:
            sim.set_params(T_inf=800,p_inf=5.0e4)
        elif t<2000.0:
            sim.set_params(T_inf=800,p_inf=3.0e7)
        elif t<3000.0:
            sim.set_params(T_inf=400,p_inf=3.0e7)
        elif t<4000.0:
            sim.set_params(T_inf=400,p_inf=5.0e3)

            
problems = dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))

# problems ={
#     'Linear_Liquid':Linear_Liquid,
#     'Small_Liquid':Small_Liquid,
#     'Small_Gas':Small_Gas,
#     'Transition_L2G':Transition_L2G,
#     'Cycle_sgclg':Cycle_sgclg,
#     'Hot_Gas':Hot_Gas,
#     }

hub = "/Users/afq/Google Drive/networks/"

import numpy as np
import os
from latent_sim import LatentSim

from SimDataDB import SimDataDB

eoses = {
    'water_slgc_logp_64':dict(
        scale_file = "data_files/water_iapw_logp_ranges.csv",
        logp=True,
        problem_list=problems.keys() # TODO watch it!
    ),
    'water_lg':dict(
        scale_file = "data_files/surf_ranges.csv",
        logp=False,
        problem_list=['Small_Liquid','Small_Gas','Hot_Gas','Transition_L2G',]
    ),
    'water_linear':dict(
        scale_file = "data_files/water_linear_ranges.csv",
        logp=False,
        problem_list=['Linear_Liquid',]
    ),
}



def perform_tests_for_eos(eos, result_dir='.'):

    networks = os.listdir(hub+'/training_'+eos)
    problem_list = eoses[eos]['problem_list']
    scale_file = eoses[eos]['scale_file']
    logp = eoses[eos]['logp']
    
    sdb = SimDataDB(result_dir+'{0}_testing.db'.format(eos))
    
    @sdb.Decorate(eos,[('problem','string'),('network','string')],
                 [('series','array')],memoize=True)
    def solve_a_problem(problem_name, network):
        print("Testing {0}:{1} on {2}".format(eos,network,problem_name))
        problem = problems[problem_name]
        ls = LatentSim(hub+'training_'+eos+'/'+network,scale_file,logp)
        q0 = ls.find_point(**problem.initial)
        ls.set_params(**problem.params)
        time_series = ls.integrate(problem.t_max, q0, schedule=problem.schedule)
        return {'series':time_series}
    
    for n in networks:
        for p in problem_list:
            solve_a_problem(p,n)

if __name__=="__main__":
    for k in eoses:
        perform_tests_for_eos(k, hub+'test_databases/')
