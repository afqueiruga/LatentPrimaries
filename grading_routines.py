from __future__ import division
from __future__ import print_function
import os, glob, re
from SimDataDB import SimDataDB

from batch_test_latent_sim import problems as test_problems

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
        for p, in problems:
            res = sdb.Query(
                'select series,run_time from {0} where problem="{1}" and network="{2}"'.
                format(eos_name,p,n))
            series,run_time = res[0]
            end_time = series[0,-1]
            if end_time >= test_problems[p].t_max - 1.0e-12:
                successes += 1
            else:
                redflag = True
            total_run_time += run_time
        summaries[n] = [ successes, total_run_time, redflag]
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
            print('{3} | {0: <32} |{1:8d} | {2:8f}'.format(
                n,v[0],v[1],'x' if v[2] else 'o'))
        print("\n")