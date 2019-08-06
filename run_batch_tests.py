import test_cfg
import directory_parsing

def job(eos):
    import batch_test_latent_sim as bt
    
    cfg = test_cfg.find_eos_test_cfg(eos)
    for problem in cfg['problem_list']:
        bt.solve_a_problem(problem, eos, result_dir=directory_parsing.test_dir)
    
if __name__=='__main__':
    # Get a list of eoses in the hubs without importing anything
    #eoses = list_eos_directories.get_list_of_eoses()
    job('water_iapws_slgc_logp')