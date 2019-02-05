import numpy as np
import sympy
from sympy import log, exp
R = 0.461526e3

triple_point   = T_t,p_t = (273.16, 611.657)
critical_point = T_c,p_c = (647.096, 22.064e6)
rho_critical   = rho_c = 322.0

import sympy
def density_enthalpy(gibbs):
    T,p = sympy.symbols('T p')
    g = gibbs(T,p)
    density = 1/g.diff(p)
    enthalpy = g - T * g.diff(T)
    rhovec = np.vectorize(sympy.lambdify([T,p],density))
    hvec = np.vectorize(sympy.lambdify([T,p],enthalpy))
                   
    return (lambda x,y : np.real(rhovec(x,y))),\
           (lambda x,y : np.real(  hvec(x,y)))

def gibbs_region1(T,p):
    p1_star = 1.653e7
    T1_star  = 1.386e3
    n1 = [ 0.14632971213167e00, -0.84548187169114e00,
          -3.7563603672040e+00,  3.3855169168385e+00, 
          -0.95791963387872e00,  0.15772038513228e00,
          -1.6616417199501e-02,  8.1214629983568e-04, 
           2.8319080123804e-04, -6.0706301565874e-04,
          -1.8990068218419e-02, -3.2529748770505e-02, 
          -2.1841717175414e-02, -5.2838357969930e-05,
          -4.7184321073267e-04, -3.0001780793026e-04, 
           4.7661393906987e-05,
          
          -4.4141845330846e-06,
          -7.2694996297594e-16, -3.1679644845054e-05, 
          -2.8270797985312e-06, -8.5205128120103e-10,
          -2.2425281908000e-06, -6.5171222895601e-07, 
          -1.4341729937924e-13, -4.0516996860117e-07,
          -1.2734301741641e-09, -1.7424871230634e-10, 
          -6.8762131295531e-19,  1.4478307828521e-20,
           2.6335781662795e-23, -1.1947622640071e-23, 
           1.8228094581404e-24, -9.3537087292458e-26]
    i1 = [ 0, 0,0, 0, 0, 0, 0, 0, 1,1, 1, 1,1,   
        1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4,4, 5,    
            8,  8, 21, 23, 29, 30, 31, 32  ]
    j1 = [ -2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0,1,
            3, -3, 0, 1,3, 17, -4, 0, 6, -5, -2, 10, -8,   
          -11, -6, -29, -31, -38, -39, -40, -41  ]
    p_i = p/p1_star
    t_i = T1_star/T
    return R*T*sum([ n*(7.1-p_i)**I*(t_i-1.222)**J
            for n,I,J in zip(n1,i1,j1)])
density_region1,enthalpy_region1 = density_enthalpy(gibbs_region1)

p1_star = 1.653e7
T1_star = 1.386e3

n1_par = [ 0.14632971213167e00, -0.84548187169114e00, -3.7563603672040e+00,  3.3855169168385e+00,
   -0.95791963387872e00,  0.15772038513228e00, -1.6616417199501e-02,  8.1214629983568e-04,
   2.8319080123804e-04, -6.0706301565874e-04, -1.8990068218419e-02, -3.2529748770505e-02,
   -2.1841717175414e-02, -5.2838357969930e-05, -4.7184321073267e-04, -3.0001780793026e-04,
   4.7661393906987e-05, -4.4141845330846e-06, -7.2694996297594e-16, -3.1679644845054e-05,
   -2.8270797985312e-06, -8.5205128120103e-10, -2.2425281908000e-06, -6.5171222895601e-07,
   -1.4341729937924e-13, -4.0516996860117e-07, -1.2734301741641e-09, -1.7424871230634e-10,
   -6.8762131295531e-19,  1.4478307828521e-20,  2.6335781662795e-23, -1.1947622640071e-23,
    1.8228094581404e-24, -9.3537087292458e-26 ]

i1_par = [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,
   1,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,
   8,  8, 21, 23, 29, 30, 31, 32 ]

i1_exp = [-1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,
   0,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,
   7,  7, 20, 22, 28, 29, 30, 31 ]

j1_par = [ -2,  -1,   0,   1,   2,   3,   4,   5,  -9,  -7,  -1,   0,   1,
    3,  -3,   0,   1,   3,  17,  -4,   0,   6,  -5,  -2,  10,  -8,
   -11,  -6, -29, -31, -38, -39, -40, -41 ]

j1_exp = [ -3,  -2,  -1,   0,   1,   2,   3,   4, -10,  -8,  -2,  -1,   0,
    2,  -4,  -1,   0,   2,  16,  -5,  -1,   5,  -6,  -3,   9,  -9,
   -12,  -7, -30, -32, -39, -40, -41, -42 ]





p2_star = 1.0e6;

T2_star = 5.4e2;

n2_ideal = [ -9.6927686500217e+00,  1.0086655968018e+01, -5.6087911283020e-03,
    7.1452738081455e-02, -4.0710498223928e-01,  1.4240819171444e+00,
    -4.3839511319450e+00, -0.28408632460772e00,  2.1268463753307e-02 ]

n2_res = [ -1.7731742473213e-03, -1.7834862292358e-02, -4.5996013696365e-02, -5.7581259083432e-02,
   -5.0325278727930e-02, -3.3032641670203e-05, -1.8948987516315e-04, -3.9392777243355e-03,
   -4.3797295650573e-02, -2.6674547914087e-05,  2.0481737692309e-08,  4.3870667284435e-07,
   -3.2277677238570e-05, -1.5033924542148e-03, -4.0668253562649e-02, -7.8847309559367e-10,
   1.2790717852285e-08,  4.8225372718507e-07,  2.2922076337661e-06, -1.6714766451061e-11,
   -2.1171472321355e-03, -2.3895741934104e+01, -5.9059564324270e-18, -1.2621808899101e-06,
   -3.8946842435739e-02,  1.1256211360459e-11, -8.2311340897998e+00,  1.9809712802088e-08,
   1.0406965210174e-19, -1.0234747095929e-13, -1.0018179379511e-09, -8.0882908646985e-11,
   1.0693031879409e-01, -3.3662250574171e-01,  8.9185845355421e-25,  3.0629316876232e-13,
   -4.2002467698208e-06, -5.9056029685639e-26,  3.7826947613457e-06, -1.2768608934681e-15,
   7.3087610595061e-29,  5.5414715350778e-17, -9.4369707241210e-07 ]



j2_ideal  = [  0,  1, -5, -4, -3, -2, -1,  2,  3 ]

i2_res = [ 1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,
            3,  4,  4,  4,  5,  6,  6,  6,  7,  7,  7,  8,  8,  9,
            10, 10, 10, 16, 16, 18, 20, 20, 20, 21, 22, 23, 24, 24, 24 ]

j2_res = [ 0,  1,  2,  3,  6,  1,  2,  4,  7, 36,  0,  1,  3,  6,
   35,  1,  2,  3,  7,  3, 16, 35,  0, 11, 25,  8, 36, 13,
   4, 10, 14, 29, 50, 57, 20, 35, 48, 21, 53, 39, 26, 40, 58 ]



    

def gibbs_region2(T,p):
    _p = p/p2_star
    _t = T2_star / T

    gamma_o = log(_p) + sum([n2_ideal[i]*_t**j2_ideal[i] for i in range(len(n2_ideal)) ])
    gamma_r = sum([
        n2_res[i] * _p**i2_res[i] * (_t-0.5)**j2_res[i]
        for i in range(len(n2_res))
        ])

    return R*T* (gamma_o + gamma_r)

density_region2,enthalpy_region2 = density_enthalpy(gibbs_region2)


def boundary_region23(T):
    return 1.0e6*(0.34805185628969e3 -0.11671859879975e1*T + 0.10192970039326e-2*T**2)



iIJ3 = [
(1,None,None),
(2,0,0),
(3,0,1),
(4,0,2),
(5,0,7),
(6,0,10),
(7,0,12),
(8,0,23),
(9,1,2),
(10,1,6),
(11,1,15),
(12,1,17),
(13,2,0),
(14,2,2),
(15,2,6),
(16,2,7),
(17,2,22),
(18,2,26),
(19,3,0),
(20,3,2),
(21,3,4),
(22,3,16),
(23,3,26),
(24,4,0),
(25,4,2),
(26,4,4),
(27,4,26),
(28,5,1),
(29,5,3),
(30,5,26),
(31,6,0),
(32,6,2),
(33,6,26),
(34,7,2),
(35,8,26),
(36,9,2),
(37,9,26),
(38,10,0),
(39,10,1),
(40,11,26),
]
ni3 = [
 0.10658070028513e1,
-0.15732845290239e2,
 0.20944396974307e2,
-0.76867707878716e1,
 0.26185947787954e1,
-0.28080781148620e1,
 0.12053369696517e1,
-0.84566812812502e-2,
-0.12654315477714e1,
-0.11524407806681e1,
 0.88521043984318,
-0.64207765181607,
 0.38493460186671,
-0.85214708824206,
 0.48972281541877e1,
-0.30502617256965e1,
 0.39420536879154e-1,
 0.12558408424308,
-0.27999329698710,
 0.13899799569460e1,

-0.20189915023570e1,
-0.82147637173963e-2,
-0.47596035734923,
 0.43984074473500e-1,
-0.44476435428739,
 0.90572070719733,
 0.70522450087967,
 0.10770512626332,
-0.32913623258954,
-0.50871062041158,
-0.22175400873096e-1,
 0.94260751665092e-1,
 0.16436278447961,
-0.13503372241348e-1,
-0.14834345352472e-1,
 0.57922953628084e-3,
 0.32308904703711e-2,
 0.80964802996215e-4,
-0.16557679795037e-3,
-0.44923899061815e-4,
]

def helmholtz_region3(T,rho):
    rho_star = rho_c
    T_star = T_c
    delta = rho / rho_star
    tau = T_star / T
    f = ni3[0] * log(delta) + sum([
        ni*delta**Ii * tau**Ji 
        for ni,(_,Ii,Ji) in zip(ni3[1:],iIJ3[1:])
    ])
    return R*T * f

def pressure_enthalpy_from_helmholtz(f_func):
    rho,T = sympy.symbols('rho T')
    f = f_func(T,rho)
    p = rho**2 * f.diff(rho)
    h = f - T*f.diff(T) + rho*f.diff(rho)
    return np.vectorize(sympy.lambdify([T,rho],p)),\
            np.vectorize(sympy.lambdify([T,rho],h))
    
    
    

def saturation_temperature(p):
    ni = [0.0e0,
    1.1670521452767e+03, -7.2421316703206e+05,
    -1.7073846940092e+01,  1.2020824702470e+04,
    -3.2325550322333e+06,  1.4915108613530e+01,
    -4.8232657361591e+03,  4.0511340542057e+05,
    -2.3855557567849e-01,  6.5017534844798e+02 ]
    p_star = 1.0e6
    T_star = 1.0
    beta = (p/p_star)**0.25
    
    E = beta**2        + ni[3]*beta + ni[6]
    F = ni[1]*beta**2 + ni[4]*beta + ni[7]
    G = ni[2]*beta**2  + ni[5]*beta + ni[8]
    D = 2.0*G/(-F-np.sqrt(F**2-4.0*E*G))
    
    return 0.5*T_star * (
        ni[10]+D-np.sqrt( (ni[10]+D)**2 - 4.0*(ni[9]+ni[10]*D) ) )