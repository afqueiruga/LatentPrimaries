from __future__ import division
from __future__ import print_function
import os, glob, re
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from SimDataDB import SimDataDB

from batch_test_latent_sim import problems as test_problems

def extract_scalars(directory):
    # TODO: check if its a top-directory or the tf training session
    archs = os.listdir(directory)
    arch_dirs = [os.path.join(directory,a) for a in archs]
    # Load summaries
    summaries = [EventAccumulator(a).Reload() for a in arch_dirs]
    # of
    tags = summary_iterators[0].Tags()['scalars']
    
    for arch,summary in zip(archs,summaries):
        pass
    
def grade_simulations(database,eos_name):
    """Examine and distill the results for each of the architectures"""
    sdb = SimDataDB(database)
    
    problems = sdb.Query('select distinct problem from {0}'.format(eos_name))
    networks = sdb.Query('select distinct network from {0}'.format(eos_name))

    summaries = {}
    for n, in networks:
        successes = 0
        total_run_time = 0.0
        redflag = False
        min_error = 1.0e5
        max_error = 0.0
        for p, in problems:
            ans = test_problems[p].answer
            ans_n = np.array([ans['T'],ans['p'], ans['rho'],ans['h']])
            res = sdb.Query(
                'select series,run_time from {0} where problem="{1}" and network="{2}"'.
                format(eos_name,p,n))
            series,run_time = res[0]
            end_time = series[-1,0]
            if end_time >= test_problems[p].t_max - 1.0e-12:
                err = np.abs( (series[-1,3:7] - ans_n)/ans_n )
#                 print(series[-1,3:7])
#                 print(ans_n)
#                 print(err)
                max_e,min_e = err.max(),err.min()
                max_error = max(max_error,max_e)
                min_error = min(min_error,min_e)
#                 if err.max()<1.0e-1:
                successes += 1
            else:
                redflag = True
            total_run_time += run_time
        summaries[n] = [ successes, total_run_time, redflag, min_error, max_error]
    return summaries

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
        test_db = hub+'test_databases/'+eos+'_testing.db'
        grades = grade_simulations(test_db,eos)
        
        
        print("Grading ",eos,":")
        for n in grades:
            v = grades[n]
            print('{flag} | {name: <32} |{successes:3d} | {min_error:1.2e} | {max_error:1.2e} | {run_time: 3.1f}'.format(
                name=n,
                successes=v[0],
                run_time=v[1],
                flag='x' if v[2] else 'o',
                min_error=v[3],
                max_error=v[4]))
        print("\n")