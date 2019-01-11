from __future__ import print_function
import numpy as np
import os
from latent_sim import LatentSim

hub = "/Users/afq/Google Drive/networks/training_water_slgc_logp_64/"
scale_file = "data_files/water_iapw_logp_ranges.csv"
logp=True


# networks = ["Classifying_2,6,24,48,sigmoid",
#             "Classifying_2,6,18,36,sigmoid",
#             "Classifying_2,5,12,24,sigmoid",
#            ]
networks = os.listdir(hub)
print(networks)
problems = []

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

problems ={
    'Small_Liquid':Small_Liquid,
    'Small_Gas':Small_Gas,
    'Transition_L2G':Transition_L2G,
    'Cycle_sgclg':Cycle_sgclg,
    }

from SimDataDB import SimDataDB

sdb = SimDataDB('testing.db')
@sdb.Decorate('water_iapw',[('problem','string'),('network','string')],
             [('series','array')],memoize=True)
def solve_a_problem(problem_name, network):
    problem = problems[problem_name]
    ls = LatentSim(hub+network,scale_file,logp)
    q0 = ls.find_point(**problem.initial)
    ls.set_params(**problem.params)
    time_series = ls.integrate(problem.t_max, q0, schedule=problem.schedule)
    return {'series':time_series}

if __name__=="__main__":
    for n in networks:
        for p in problems:
            solve_a_problem(p,n)
