import sys, inspect
from equations_of_state import iapws97

# TODO the meaning of the "answers" has changed now

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
    
class Linear_Liquid_Heat():
    t_max = 10.0
    initial = dict(p = 1.0e5, T = 20.0+273.15, phase="Liquid")
    params =  dict(k_p=1.0e-4,k_T=1.0e6)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=294.15,p_inf=1.0e5)
    answer = dict(T=294.15,p=1.0e5,
        rho=iapws97.density_region1(294.15,1.0e5),
        h  =iapws97.enthalpy_region1(294.15,1.0e5) )
    
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
    params =  dict(k_p=1.0e-4,k_T=-1.0e4,Dt=10.0)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=450,p_inf=5.0e3)
    answer = dict(T=450,p=5.0e3,
        rho= iapws97.density_region2(450.0,5.0e3),
        h  =iapws97.enthalpy_region2(450.0,5.0e3) )

class Hot_Gas():
    t_max = 1000.0
    initial = dict(T=450,p=5.0e5, phase="Gas")
    params =  dict(k_p=1.0e-4,k_T=1.0e4, Dt=100.0)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=550,p_inf=5.0e5)
    answer = dict(T=550,p=5.0e5,
        rho= iapws97.density_region2(550.0,5.0e5),
        h  =iapws97.enthalpy_region2(550.0,5.0e5) )

class Hot_Gas_Fill():
    t_max = 1000.0
    initial = dict(T=450,p=5.0e5, phase="Gas")
    params =  dict(mass_source=0.1,k_p=0.0,k_T=0.0, Dt=t_max/1000.0)
    @staticmethod
    def schedule(sim,t):
        pass
    answer = dict(T=450,p=5.0e5,
        rho= iapws97.density_region2(450.0,5.0e5),
        h  =iapws97.enthalpy_region2(450.0,5.0e5) )
    
class Transition_L2G():
    t_max = 10.0
    initial = dict(T=350,p=5.0e5, phase="Liquid")
    params =  dict(k_p=1.0e-2,k_T=1.0e7, Dt=t_max/1000.0)
    @staticmethod
    def schedule(sim,t):
        sim.set_params(T_inf=350,p_inf=5.0e3)
    answer = dict(T=350,p=5.0e3,
        rho= iapws97.density_region2(350.0,5.0e3),
        h  =iapws97.enthalpy_region2(350.0,5.0e3) )
    
class Transition_L2G_Drain():
    t_max = 10.0
    initial = dict(T=350,p=5.0e5, phase="Liquid")
    params =  dict(mass_source=-0.1,k_p=0.0,k_T=0.0, Dt=t_max/1000.0)
    @staticmethod
    def schedule(sim,t):
        pass
    answer = dict(T=350,p=5.0e3,
        rho= iapws97.density_region2(350.0,5.0e3),
        h  =iapws97.enthalpy_region2(350.0,5.0e3) )
    
class Liquid_Drain():
    t_max = 2000.0
    initial = dict(T=373.15,p=3.0e5, phase="Liquid")
    params =  dict(mass_source=-0.1,k_p=0.0,k_T=0.0, Dt=t_max/1000.0)
    @staticmethod
    def schedule(sim,t):
        pass
    answer = dict(T=350,p=5.0e3,
        rho= iapws97.density_region2(350.0,5.0e3),
        h  =iapws97.enthalpy_region2(350.0,5.0e3) )
    
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
    

all_test_problems = dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))


# This matches 
eos_test_cfg = {
    '_slgc':dict(
        scale_file = "data_files/water_iapws_logp_ranges.csv",
        logp=True,
        problem_list=all_test_problems.keys() # TODO watch it!
    ),
    '_lg':dict(
        scale_file = "data_files/water_iapws_lg_ranges.csv",
        logp=False,
        problem_list=['Small_Liquid','Small_Gas','Hot_Gas','Transition_L2G',
                      'Transition_L2G_Drain'
                     'Liquid_Drain','Hot_Gas_Fill']
    ),
    '_linear':dict(
        scale_file = "data_files/water_linear_ranges.csv",
        logp=False,
        problem_list=['Linear_Liquid',]
    ),
}
def find_eos_test_cfg(key):
    for dataset_match in eos_test_cfg:
        if dataset_match in key:
            return eos_test_cfg[dataset_match]

