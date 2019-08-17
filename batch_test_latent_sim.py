from __future__ import print_function
import numpy as np
import os
from SimDataDB import SimDataDB


from latent_sim import LatentSim
from eoshub import EOSHub

from test_cfg import *

def curried_latentssim(eos,network):
    """Curried constructor for LatentSim with the hub and 
    eos registry."""
    scale_file = eoses[eos]['scale_file']
    logp = eoses[eos]['logp']
    ls = LatentSim(hub+'training_'+eos+'/'+network,scale_file,logp)
    return ls


def solve_a_problem_arch(problem_name, eos, network=None):
    problem = all_test_problems[problem_name]
    ls = EOSHub.LatentSim(eos,network)
    q0 = ls.find_point(**problem.initial)
    ls.set_params(**problem.params)
    time_series = ls.integrate(problem.t_max, q0, 
                               schedule=problem.schedule,verbose=False)
    return time_series    

def solve_a_problem(problem_name,eos, result_dir='.'):
    sdb = SimDataDB(result_dir+'/{0}_testing.db'.format(eos))
    @sdb.Decorate(eos,[('problem','string'),('network','string')],
                      [('series','array')],memoize=False)
    def _solve(problem_name,network):
        print("Testing {0}:{1} on {2}".format(eos,network,problem_name))
        time_series = solve_a_problem_arch(problem_name,eos,network)
        return {'series':time_series}
    def _job(arch):
        try:
            _solve(problem_name,arch)
        except Exception as e:
            print(f"The network {arch} threw an error:\n    {e}")
    for arch in EOSHub[eos].archs:
        #_solve(problem_name,arch)
        _job(arch)

#
# Deprecated scripting before EOSHub
#
def DEP_run_one_simulation(eos,network,problem_name,verbose=True):
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
    
def DEP_perform_tests_for_eos(eos, result_dir='.'): # dep
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
        try:
            for p in problem_list:
                solve_a_problem(p,n)
        except Exception as e:
            print("The network", n, " threw an error: ", e)
            
# This is now a library; batching requires clever forking before imports
# if __name__=="__main__":
#     for k in eoses:
#         try:
#             perform_tests_for_eos(k, hub+'test_databases/')
#         except FileNotFoundError:
#             pass

