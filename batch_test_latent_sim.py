from __future__ import print_function
import sys,inspect
import iapws97

class Linear_Liquid():
    t_max = 1.0
    initial = dict(p = 1.0e5, T = 20.0+273.15, phase="Liquid")
    params =  dict(k_p=1.0e-4,k_T=1.0e4)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=293.15,p_inf=1.0e5+10.0)
    answer = dict(T=293.15,p=1.0e5+10.0,
        rho=iapws97.density_region1(293.15,1.0e5+10.0),
        h  =iapws97.enthalpy_region1(293.15,1.0e5+10.0) )
    
class Small_Liquid():
    t_max = 1000.0
    initial = dict(T=350,p=5.0e5, phase="Liquid")
    params =  dict(k_p=1.0e-4,k_T=1.0e4,Dt=10.0)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=450,p_inf=5.0e5)
    answer = dict(T=450,p=5.0e5,
        rho=iapws97.density_region1(450.0,5.0e5),
        h  =iapws97.enthalpy_region1(450.0,5.0e5) )

class Small_Gas():
    t_max = 1000.0
    initial = dict(T=350,p=5.0e3, phase="Gas")
    params =  dict(k_p=1.0e-4,k_T=1.0e4,Dt=10.0)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=450,p_inf=5.0e3)
    answer = dict(T=450,p=5.0e3,
        rho= iapws97.density_region2(450.0,5.0e3),
        h  =iapws97.enthalpy_region2(450.0,5.0e3) )

class Hot_Gas():
    t_max = 10000.0
    initial = dict(T=450,p=5.0e5, phase="Gas")
    params =  dict(k_p=1.0e-4,k_T=1.0e4, Dt=100.0)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=550,p_inf=5.0e5)
    answer = dict(T=550,p=5.0e5,
        rho= iapws97.density_region2(450.0,5.0e5),
        h  =iapws97.enthalpy_region2(450.0,5.0e5) )
        
class Transition_L2G():
    t_max = 1.0
    initial = dict(T=350,p=5.0e5, phase="Liquid")
    params =  dict(k_p=1.0e-4,k_T=1.0e4, Dt=t_max/100.0)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=350,p_inf=5.0e3)
    answer = dict(T=350,p=5.0e3,
        rho= iapws97.density_region2(450.0,5.0e3),
        h  =iapws97.enthalpy_region2(450.0,5.0e3) )
    
class Cycle_sgclg():
    t_max = 100.0
    initial = dict(T=250,p=5.0e3, phase="Solid")
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
    answer = dict(T=400,p=5.0e3,
        rho= iapws97.density_region2(400.0,5.0e3),
        h  =iapws97.enthalpy_region2(400.0,5.0e3) )
    

problems = dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))


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

def run_one_simulation(eos,network,problem_name,verbose=True):
    """Run one of the tests in an environment we can embed into."""
    scale_file = eoses[eos]['scale_file']
    logp = eoses[eos]['logp']
    problem = problems[problem_name]
    ls = LatentSim(hub+'training_'+eos+'/'+network,scale_file,logp)
    q0 = ls.find_point(**problem.initial)
    ls.set_params(**problem.params)
    time_series = ls.integrate(problem.t_max, q0, 
                               schedule=problem.schedule,
                              verbose=verbose)
    return time_series, ls
    
def perform_tests_for_eos(eos, result_dir='.'):
    """Perform all of the tests and generate a report."""
    networks = os.listdir(hub+'/training_'+eos)
    problem_list = eoses[eos]['problem_list']
    scale_file = eoses[eos]['scale_file']
    logp = eoses[eos]['logp']
    
    sdb = SimDataDB(result_dir+'{0}_testing.db'.format(eos))
    
    @sdb.Decorate(eos,[('problem','string'),('network','string')],
                 [('series','array')],memoize=False)
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
