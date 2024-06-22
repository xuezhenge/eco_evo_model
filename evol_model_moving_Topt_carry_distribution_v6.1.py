from sympy import *
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import norm
from joblib import Parallel, delayed
import time
from sympy import symbols, Eq, solve
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--num_cores', type=int,
    default=6)
parser.add_argument('--num_years', type=int,
    default=150)
parser.add_argument('--climate', type=str,
    default='smooth')
parser.add_argument('--dt', type=float,
    default=0.5)
parser.add_argument('--fn_id', type=int,
    default=0)
parser.add_argument('--vG_A0', type=float,
    default=0.3, help='0,0.075,0.15,0.3,0.6,1.2')
parser.add_argument('--vG_L0', type=float,
    default=0.3,help='0,0.075,0.15,0.3,0.6,1.2')
parser.add_argument('--vE_A', type=float,
    default=0.7, help='0,0.175,0.35,0.7,1.4,2.8')
parser.add_argument('--vE_L', type=float,
    default=0.7)
parser.add_argument('--ov_mor', type=float,
    default=0.2)
parser.add_argument('--num_steps', type=int,
    default=150)
parser.add_argument('--task', type=str,
    default='SA')
parser.add_argument('--model', type=str,
    default='C+P+G')
parser.add_argument('--seas', type=float,
    default=9.942)
parser.add_argument('--trend', type=float,
    default=5.793)
parser.add_argument('--Fis', type=float,
    default=0.1)
parser.add_argument('--test_year', type=int,
    default=1)
parser.add_argument('--approach', type=str,
    default='carry_distribution')
 
code_version = 'v6.1'

args = parser.parse_args()
num_cores = args.num_cores
num_years = args.num_years
dt = args.dt
climate = args.climate
ov_mor = args.ov_mor
fn_id = args.fn_id
num_steps = args.num_steps
task = args.task
model = args.model
seas = args.seas
trend = args.trend
Fis = args.Fis
test_year = args.test_year
approach = args.approach

#vG_A0 = args.vG_A0; vG_L0 = args.vG_L0; vE_A = args.vE_A; vE_L = args.vE_L
# parameters
q1A = 1.5
K_A = 50000000

# parameters for predation rate
a = 0.000001*0.2
h = 0.01
mL = 1.0
q1L = 1.5

q2A = 1
q2L = 1
min_diff = -15
max_diff = 10
Topt0 = 25

def writer_csv(rows, filename,header):
    with open(filename, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def day_length(day_of_year,lat):
    # https://www.sciencedirect.com/science/article/pii/S0304380016301946#bib0050
    # k: the exposed radius between the sun’s zenith and the sun’s solar circle
    # R: the Earth’s rotational axis, R = 23.439°
    pi = math.pi
    R = 23.439
    k = math.tan(pi * lat / 180) * math.tan(( pi * R / 180) * math.cos( pi* day_of_year / 182.625))
    if k < -1:
        k_ = -1
    elif k > 1:
        k_ = 1
    else:
        k_ = k
    DL = (24 / pi) * math.acos(k_)
    return DL

def dailytemp(fn,seas,trend):
    if fn == '-84.0_32.0.csv':
        Tmean = 15.973
    elif fn == '-84.0_38.0.csv':
        Tmean = 10.723
    elif fn == '-84.0_44.0.csv':
        Tmean = 6.477
    elif fn == '-94.0_32.0.csv':
        Tmean = 16.889
    elif fn == '-94.0_38.0.csv':
        Tmean = 11.252
    elif fn == '-94.0_44.0.csv':
        Tmean = 5.103
    elif fn == '-104.0_32.0.csv':
        Tmean = 13.318
    elif fn == '-104.0_38.0.csv':
        Tmean = 5.055
    elif fn == '-104.0_44.0.csv':
        Tmean = 4.153
    #import pdb;pdb.set_trace()
    ts = np.arange(365*150)
    temps = []
    for t in ts:
        temp = -seas*math.cos(2*math.pi*t/365) + Tmean + trend*t/36500
        temps.append(temp)
    temps = pd.DataFrame(np.array(temps),columns = ['Temp'])
    temp_data = np.array(temps.Temp).reshape((150,365))
    return temp_data

# temperature performance curve
def thorneley_france(m,Topt,q1,q2,T,deltaE_A):
    Topt = Topt + deltaE_A
    Tmin = Topt + min_diff
    Tmax = Topt + max_diff
    # Generic temperature-dependent function
    if (T>=(Tmin)) & (T<=(Tmax)):
        return m*(((T-(Tmin))**q1)*((Tmax-T)**q2))/(((Topt-Tmin)**q1)*((Tmax-Topt)**q2))
    else:
        return 0.0

# food dependent predation rate
def fdpr(a,h,A):
    return a/(1+a*h*A)

#all years daily temperature function
def get_temps(lon,lat):
    if task == 's0t1' or task == 's1t0':
        temp_df = pd.read_csv(f'../../inputs/{climate}_temp_{task}/{lon}_{lat}.csv')
    else:
        temp_df = pd.read_csv(f'../../inputs/{climate}_temp/{lon}_{lat}.csv')
    temp_data = np.array(temp_df.Temp).reshape((150,365))
    return temp_data

def get_daily_temp(t,year,temp_data):
    t = int(np.floor(t))
    if t == 365:
        Temp_t = temp_data[year-1,:][364]
    else:
        Temp_t = temp_data[year-1,:][t]
    return Temp_t

def interpolated_temp(t,year,temp_data):
    #integrate temperature at each dt
    if model == 'NC+NP+NG':
        year = 1
    if model == 'NC+P+NG' and year > 20:
        year =20
    temp = temp_data[year-1,:]
    temp_len = len(temp)
    t_tmp = int(np.floor(t))
    if t_tmp <= (temp_len-1):
        Temp_t = temp[t_tmp-1] + (temp[t_tmp] - temp[t_tmp-1])*(t - t_tmp)
    else:
        Temp_t = temp[temp_len-1]
    return Temp_t

#switch from asexual to sexual phrase (with overwintering period)
def switch_sexual(t,year,temp_data,Topt,lat):
    Temp_t = interpolated_temp(t,year,temp_data)
    Tmin = Topt + min_diff
    Tmax = Topt + max_diff
    DL =  day_length(t,lat)
    if t <  182.5:
        switch = 0
    elif t >= 182.5:
        if Temp_t >= Topt:
            switch = 0
        elif Temp_t <= Tmin or DL <= -0.05*Temp_t + 15.67:
            switch = 1
        else: switch = 0
    return switch

# temperature dependent mortality rate
def get_paras_mor2(Tmin,Tmax,Topt1,Topt2,v_max,v_min):
    # tic = time.time()
    a1,a2,b1,b2 = symbols('a1 a2 b1 b2')
    eq1 = Eq(a1*Tmin + b1, v_max)
    eq2 = Eq(a1*Topt1 + b1, v_min)
    eq3 = Eq(a2*Tmax + b2, v_max)
    eq4 = Eq(a2*Topt2 + b2, v_min)
    roots = solve((eq1,eq2,eq3,eq4),(a1,a2,b1,b2))
    a1 = roots.get(a1)
    a2 = roots.get(a2)
    b1 = roots.get(b1)
    b2 = roots.get(b2)
    # print(f"eta: {time.time() - tic}")
    return a1,a2,b1,b2

def mor_rate(T,a1,a2,b1,b2, Topt,deltaE_A,v_min,v_max):
    # the function of temperature dependent mortality rate
    Topt = Topt + deltaE_A
    Tmin = Topt + min_diff
    Tmax = Topt + max_diff
    Topt1 = Topt - 5
    Topt2 = Topt + 5
    shift = Topt-Topt0
    if (T >= Tmin) & (T <= Topt1):
        rate = a1*(T - shift) + b1
    elif (T > Topt1) & (T < Topt2):
        rate = v_min
    elif (T >= Topt2) & (T <= Tmax):
        rate = a2*(T - shift) + b2
    else: rate = v_max
    return rate

def morAL(Temp_t_cur,Topt,deltaE_A,idx):
    # aphid: v_min = 0.05, v_max = 0.25
    # ladybird: v_min = 0.01, v_max = 0.1
    #a1L,a2L,b1L,b2L = get_paras_mor2(Tmin,Tmax,Topt_mor1,Topt_mor2,v_max,v_min)
    if idx == 'aphid':
        a1 = -0.02; b1 = 0.45; a2 = 0.04; b2 = -1.15
        mor = mor_rate(Temp_t_cur, a1, a2, b1, b2, Topt, deltaE_A,v_min = 0.05, v_max = 0.25)
    elif idx == 'ladybird':
        a1 = -0.009; b1 = 0.19; a2 = 0.018; b2 = -0.53
        mor = mor_rate(Temp_t_cur, a1, a2, b1, b2, Topt, deltaE_A,v_min = 0.01, v_max = 0.1)
    return mor

def temp_dependent_conversion_factor(m,Topt,q1,q2,T,r_min,r_max):
    Tmin = Topt + min_diff
    Tmax = Topt + max_diff
    if T>=Tmin and T<=Tmax:
        # temperature-dependent rate
        tdr = thorneley_france(m,Topt,q1,q2,T,0)
        TDCF = max(tdr*r_max,r_min)
    else:
        TDCF = 0.0
    return TDCF

def get_Qp(Topt,Temp):
    #make sure Qp_min falls into a range of 500-2000, then the range of the intrinstic rate of growth is 0.05-0.2
    Qp_min = 500
    #temperature dependent conversion_factor
    TDCF = temp_dependent_conversion_factor(mL,Topt,q1L,q2L,Temp,0.25,1)
    if TDCF == 0:
        Qp = 2000 #temperature is not suitable for ladybird, the value of Qp doesn't matter.
    else:
        Qp = 1/((1/Qp_min)*TDCF)
    return Qp

# def get_mA(Topt,Temp):
#     # make sure the range of the inrinstic of growth is 0.1 - 0.5
#     mA_max = 0.55+0.05
#     #temperature dependent conversion_factor
#     TDCF = temp_dependent_conversion_factor(mL,Topt,q1L,q2L,Temp,0.5,1)
#     mA = mA_max*TDCF
#     return mA

def get_range(v):
    #https://homepage.divms.uiowa.edu/~mbognar/applets/normal.html
    if v == 0 or v == 0.0001:
        rangeV = 0.04
    elif v == 0.001*0.7*0.1:
        rangeV = 0.01
    elif v == 0.3 or v == 0.15:
        #rangeV = 2
        rangeV = 2 #increase it to cover wider range to improve accuracy
    elif v == 0.6 or v == 0.45:
        rangeV = 3
    elif v == 0.7:
        rangeV = 3
    elif v == 0.9:
        rangeV = 4
    elif v == 1.2:
        rangeV = 4
    elif v == 2.4:
        rangeV = 8
    return rangeV

# model equations:
def Solve_euler_model(year,A_add,L_add,temp_data,lat,ToptA_end,ToptL_end,t_start, t_end,dt):
    tic = time.perf_counter()
    ts = np.arange(t_start,t_end,dt)
    n_t=len(ts)
    A = np.empty(n_t); A.fill(0)
    L = np.empty(n_t); L.fill(0)
    ToptAs = np.empty(n_t);ToptAs.fill(ToptA_end)
    ToptLs = np.empty(n_t);ToptLs.fill(ToptL_end)
    vG_A = np.empty(n_t)
    vG_L = np.empty(n_t)
    vG_A.fill(vG_A_end[year-1])
    vG_L.fill(vG_L_end[year-1])
    ToptA_fix = ToptAs[0]
    vG_A_fix = vG_A[0]
    print(vG_A[0],vG_L[0])
    dL_sv = np.zeros(n_t)
    w_A = np.zeros([n_t]);w_L = np.zeros([n_t])
    trait_distAs = list()
    # For loop to create each DataFrame and add it to the list
    for i in range(n_t):
        # Create an empty DataFrame with specified column names
        df = pd.DataFrame(columns=['ToptA_G,density,normal_density'])
        # Add the DataFrame to the list
        trait_distAs.append(df)

    num_change_A = 0; num_change_L = 0
    for i in np.arange(1,n_t):
        t = ts[i-1] #previous time step
        t_cur = ts[i] #current time step
       
        # Temperature at current time step
        Temp_t_cur = interpolated_temp(t_cur,year,temp_data)

        # Shift and vG_A at previous time step
        ToptA_t = ToptAs[i-1]; ToptL_t = ToptLs[i-1];vG_A_t = vG_A[i-1]; vG_L_t = vG_L[i-1]; dL_sv_t = dL_sv[i-1]
        vG_A_sqrt = math.sqrt(vG_A_t)
        vG_L_sqrt = math.sqrt(vG_L_t)

        #mA and Qp
        mA = 0.6
        Qp = get_Qp(ToptL_t,Temp_t_cur)

        if Temp_t_cur < ToptA_t - 15 and num_change_A == 0:
            i_start = i
            continue
        else:
            def get_AL(t,A_t,L_t):
                fdpr_effect = fdpr(a,h,A_t)
                carrying_effect = (1-A_t/K_A)
                # numerical integration for aphid and ladybird over environmental effect
                #for aphid
                def p_integA(ToptA_G,ToptA_t,ToptL_t,A_t,L_t,Temp_t_cur,beta,trait_distA_t):
                    if vE_A != 0:
                        if approach == 'carry_distribution':
                            rhoGA = trait_distA_t.loc[trait_distA_t['ToptA_G'] == ToptA_G, 'density'].values[0]
                        elif approach == 'normal_distribution':
                            rhoGA = norm.pdf(ToptA_G, loc = ToptA_t, scale = vG_A_sqrt)
                        expr = lambda deltaE_A: (thorneley_france(mA,ToptA_G,q1A,q2A,Temp_t_cur,deltaE_A)*carrying_effect - morAL(Temp_t_cur,ToptA_G,deltaE_A,idx = 'aphid'))*norm.pdf(deltaE_A, loc = 0, scale = vE_A_sqrt)
                        integrate_E = integrate.quad(expr, -rangeVEA, rangeVEA)[0]
                        numer_rhoA = rhoGA + rhoGA*integrate_E*dt - rhoGA*beta*dt
                    # else:
                    #     integrate_E = (thorneley_france(mA,ToptA_G,q1A,q2A,Temp_t_cur,0)*carrying_effect - morAL(Temp_t_cur,ToptA_G,0,idx = 'aphid'))*norm.pdf(ToptA_G, loc = ToptA_t, scale = vG_A_sqrt)
                    return numer_rhoA
                
                def p_integA(ToptA_GM,ToptA_t,ToptL_t,A_t,L_t,Temp_t_cur,beta,trait_distA_t):
                    if vE_A != 0:
                        rhoGMA = trait_distA_t.loc[trait_distA_t['ToptA_G'] == ToptA_GM, 'density'].values[0]
                        expr = lambda deltaE_A: (thorneley_france(mA,ToptA_GM,q1A,q2A,Temp_t_cur,deltaE_A)*carrying_effect - morAL(Temp_t_cur,ToptA_GM,deltaE_A,idx = 'aphid'))*norm.pdf(deltaE_A, loc = 0, scale = vE_A_sqrt)
                        integrate_E = integrate.quad(expr, -rangeVEA, rangeVEA)[0]
                        numer_rhoGA = rhoGMA*(1 + integrate_E*dt - beta*dt)
                        # add mutational variance
                        expr_rhoGM = lambda ToptA_G: numer_rhoGA*norm.pdf(ToptA_GM-ToptA_G,loc = 0, scale = math.sqrt(vmA*dt))
                        numer_rhoGM = integrate.quad(expr_rhoGM,ToptA_GM-rangeVMA,ToptA_GM+rangeVMA)[0]
                    # else:
                    #     integrate_E = (thorneley_france(mA,ToptA_G,q1A,q2A,Temp_t_cur,0)*carrying_effect - morAL(Temp_t_cur,ToptA_G,0,idx = 'aphid'))*norm.pdf(ToptA_G, loc = ToptA_t, scale = vG_A_sqrt)
                    return numer_rhoGA

                #for ladybird
                def p_integL(ToptL_G,ToptA_t,ToptL_t,A_t,L_t,Temp_t_cur):
                    if vE_L != 0:
                        rhoGL = norm.pdf(ToptL_G, loc = ToptL_t, scale = vG_L_sqrt)
                        expr = lambda deltaE_L: (fdpr_effect*thorneley_france(mL,ToptL_G,q1L,q2L,Temp_t_cur,deltaE_L)*A_t/Qp - morAL(Temp_t_cur,ToptL_G,deltaE_L,idx = 'ladybird'))*norm.pdf(deltaE_L, loc = 0, scale = vE_L_sqrt)
                        integrate_E = integrate.quad(expr, -rangeVEL, rangeVEL)[0]
                        numer_rhoL = rhoGL + rhoGL*integrate_E*dt
                    # else:
                    #     integrate_E = (fdpr_effect*thorneley_france(mL,ToptL_G,q1L,q2L,Temp_t_cur,0)*A_t/Qp - morAL(Temp_t_cur,ToptL_G,0,idx = 'ladybird'))*norm.pdf(ToptL_G, loc = ToptL_t, scale = vG_L_sqrt)
                    return numer_rhoL

                rangeVGA = get_range(vG_A0)
                rangeVEA = get_range(vE_A)
                rangeVGL = get_range(vG_L0)
                rangeVEL = get_range(vE_L)
                rangeVMA = get_range(vmA)

                stepA = rangeVGA*2/num_steps
                stepL = rangeVGL*2/num_steps
                
                ToptA_Gs = np.arange(ToptA_fix - rangeVGA-1,ToptA_fix + rangeVGA + 1 +stepA,stepA)
                ToptL_Gs = np.arange(ToptL_t - rangeVGL,ToptL_t + rangeVGL +stepL,stepL)

                if num_change_A == 0:
                    den = [norm.pdf(ToptA_G, loc = ToptA_fix, scale = math.sqrt(vG_A_fix)) for ToptA_G in ToptA_Gs]
                    trait_distA_t = pd.DataFrame({'ToptA_G': ToptA_Gs, 'density': den, 'density': den})
                    trait_distAs[i-1] = trait_distA_t
                else:
                    trait_distA_t = trait_distAs[i-1]
                # for aphid
                if vG_A0 == 0:
                    new_ToptA = ToptA_t
                    new_vG_A = vG_A0
                    w_A[i] = thorneley_france(mA,ToptA_t,q1A,q2A,Temp_t_cur,0)*carrying_effect - fdpr_effect*thorneley_france(mL,ToptL_t,q1L,q2L,Temp_t_cur,0)*L_t - morAL(Temp_t_cur,ToptA_t,0,idx = 'aphid')
                    denomA = w_A[i] + 1
                else:
                    if vG_L0 == 0:
                        new_ToptL = ToptL_t
                        new_vG_L = vG_L0
                        beta = fdpr_effect*thorneley_france(mL,ToptL_t,q1L,q2L,Temp_t_cur,0)*L_t
                        FA = thorneley_france(mA,ToptA_t,q1A,q2A,Temp_t_cur,0)*carrying_effect - morAL(Temp_t_cur,ToptA_t,0,idx = 'aphid')
                    else:
                        expr_beta = lambda ToptL_G,deltaE_L: fdpr_effect*thorneley_france(mL,ToptL_G,q1L,q2L,Temp_t_cur,deltaE_L)*L_t*norm.pdf(ToptL_G, loc = ToptL_t, scale = vG_L_sqrt)*norm.pdf(deltaE_L, loc = 0, scale = vE_L_sqrt)
                        beta = integrate.dblquad(expr_beta, -rangeVEL, rangeVEL,ToptL_t-rangeVGL, ToptL_t + rangeVGL)[0]
                    # numerical integration for aphid over genetic and environmental effect
                    resA = Parallel(n_jobs=num_cores)(delayed(p_integA)(item,ToptA_t,ToptL_t,A_t,L_t,Temp_t_cur,beta,trait_distA_t) for item in ToptA_Gs)
                    denomA = sum([resA[i]*stepA for i in range(len(resA))])
                    new_rhoA = [resA[i]/denomA for i in range(len(resA))]

                    if num_change_A == 0:
                        den = [norm.pdf(ToptA_G, loc = ToptA_fix, scale = math.sqrt(vG_A_fix)) for ToptA_G in ToptA_Gs]
                        trait_distAs[i] = pd.DataFrame({'ToptA_G': ToptA_Gs, 'density': den,'norm_density':den})
                    elif num_change_A == 1:
                        den = [norm.pdf(ToptA_G, loc = ToptA_fix, scale = math.sqrt(vG_A_fix)) for ToptA_G in ToptA_Gs]
                        trait_distAs[i] = pd.DataFrame({'ToptA_G': ToptA_Gs, 'density': new_rhoA,'norm_density':den})  

                    #range of the new distribution
                    distA_range_min = ToptA_t - rangeVGA
                    distA_range_max = ToptA_t + rangeVGA
                
                    filtered_IDs = [index for index, value in enumerate(ToptA_Gs) if distA_range_min <= value <= distA_range_max]
                    len_IDs = len(filtered_IDs)
                    stepA = rangeVGA*2/len_IDs
                    denomA = sum([resA[filtered_IDs[i]]*stepA for i in range(len_IDs)])
                    numerA = sum([ToptA_Gs[filtered_IDs[i]] * resA[filtered_IDs[i]]*stepA for i in range(len_IDs)])
                    # # new mean of the shiftA after selection
                    new_ToptA = numerA/denomA
                    # new additive genetic variance (Have already included mutational variance in new_rhoA)
                    new_vG_A = sum([(ToptA_Gs[filtered_IDs[i]]- new_ToptA)**2 * resA[filtered_IDs[i]]*stepA/denomA for i in range(len_IDs)])
                    w_A[i] = denomA - 1

                    #when switch from asexual phrase to sexual phrase
                    dA_sv[year] = (1-Fis)*dA_sv[year-1]/2 + (new_vG_A - vG_A_end[year-1])
                    new_vG_A_prime =(1-Fis)*vG_A0 + dA_sv[year]
                    vG_A_end[year] = new_vG_A_prime + vmA #aphids have mutational variance added at the random mating phase, the "full amount" because it's like a generation of mating (no delta_t). 

                # for ladybird
                if vG_L0 == 0:
                    new_ToptL = ToptL_t
                    new_vG_L = vG_L0
                    w_L[i] = fdpr_effect*thorneley_france(mL,ToptL_t,q1L,q2L,Temp_t_cur,0)*A_t/Qp - morAL(Temp_t_cur,ToptL_t,0,idx = 'ladybird')
                    denomL = w_L[i] + 1
                else:
                    resL = Parallel(n_jobs=num_cores)(delayed(p_integL)(item,ToptA_t,ToptL_t,A_t,L_t,Temp_t_cur) for item in ToptL_Gs)
                    numerL = sum([ToptL_Gs[i] * resL[i]*stepL for i in range(len(resL))])
                    denomL = sum([resL[i]*stepL for i in range(len(resL))])
                    #denomL = 1 + wL_bar*dt
                    new_ToptL = numerL/denomL
                    # new additive genetic variance due to selection (need to add mutational variance)
                    new_vG_L = sum([(ToptL_Gs[i]- new_ToptL)**2 * resL[i]*stepL/denomL for i in range(len(resL))]) + vmL*dt

                    # new additive genetic variance due to segeration variance
                    dL_sv[i] = dL_sv_t/2 + (new_vG_L - vG_L_t)
                    new_vG_L = vG_L0 + dL_sv[i]
                    vG_L_end[year] = new_vG_L

                    # print(i,vG_A_end[year],vG_L_end[year])

                    w_L[i] = denomL - 1

                if num_change_A == 0:
                    ToptAs[i] = ToptA_t
                    vG_A[i] = vG_A_t
                else:
                    ToptAs[i] = new_ToptA
                    vG_A[i] = new_vG_A
                if num_change_L == 0:
                    ToptLs[i] = ToptL_t
                    vG_L[i] = vG_L_t
                    dL_sv[i] = dL_sv_t
                else:
                    ToptLs[i] = new_ToptL
                    vG_L[i] = new_vG_L

                A[i] = A_t*denomA
                L[i] = L_t*denomL
                # print(i,Temp_t_cur,new_ToptA,new_ToptL,new_vG_A,new_vG_L,w_A[i],w_L[i],A[i],L[i])
                return w_A[i],w_L[i],A[i],L[i]

            if num_change_A == 0:
                # add aphid into the model if wA >0 constantly for 5 days)
                A_t,L_t = [A_add,0]
                def get_wA(t,A_t,L_t):
                    wA,wL,A_t,L_t = get_AL(t,A_t,L_t)
                    return wA,A_t,L_t
                
                wA1,A_t,L_t = get_wA(t,A_t,L_t)
                if wA1 > 0:
                    wA2,A_t,L_t = get_wA(t+1,A_t,L_t)
                    if wA2 >0 :
                        wA3,A_t,L_t = get_wA(t+2,A_t,L_t)
                        if wA3 > 0:
                            wA4,A_t,L_t = get_wA(t+3,A_t,L_t)
                            if wA4 > 0:
                                wA5, A_t,L_t = get_wA(t+4,A_t,L_t)
                                if wA5 >0:
                                    num_change_A = 1
                                    get_AL(t,A_add,0)
                                    t_addA = t_cur
                                    print(f'Year {year} Day {t_cur}: add aphid')
                                else:
                                    num_change_A = 0
                                    get_AL(t,0,0)
                            else:
                                num_change_A = 0
                                get_AL(t,0,0)
                        else:
                            num_change_A = 0
                            get_AL(t,0,0)
                    else:
                        num_change_A = 0
                        get_AL(t,0,0)
                else:
                    num_change_A = 0
                    get_AL(t,0,0)
            
            elif num_change_A == 1 and num_change_L == 0:
                # add ladybird into the model (wL >0 constantly for 5 days)
                A_t,L_t = [A[i-1],L_add]
                def get_wL(t,A_t,L_t):
                    wA,wL,A_t,L_t = get_AL(t,A_t,L_t)
                    return wL,A_t,L_t
                wL1,A_t,L_t = get_wL(t,A_t,L_t)
                if wL1 >0:
                    wL2,A_t,L_t = get_wL(t+1,A_t,L_t)
                    if wL2 >0:
                        wL3,A_t,L_t = get_wL(t+2,A_t,L_t)
                        if wL3>0:
                            wL4,A_t,L_t = get_wL(t+3,A_t,L_t)
                            if wL4>0:
                                wL5,A_t,L_t = get_wL(t+4,A_t,L_t)
                                if wL5 > 0:
                                    num_change_L = 1
                                    get_AL(t,A[i-1],L_add)
                                    t_addL = t_cur
                                    print(f'Year {year} Day {t_cur}: add ladybird')
                                else:
                                    num_change_L = 0
                                    get_AL(t,A[i-1],0)
                            else:
                                num_change_L = 0
                                get_AL(t,A[i-1],0)   
                        else:
                            num_change_L = 0
                            get_AL(t,A[i-1],0)
                    else:
                        num_change_L = 0
                        get_AL(t,A[i-1],0)    
                else:
                    num_change_L = 0
                    get_AL(t,A[i-1],0)

            elif num_change_A == 1 and num_change_L == 1:
                # aphid and ladybird have already been added into the model
                get_AL(t,A[i-1],L[i-1])

            if A[i] <0 or L[i] < 0:
                break
        # End the simulation
        A_end = A[i];L_end = L[i];ToptA_end = ToptAs[i]; ToptL_end = ToptLs[i]
        switch= switch_sexual(t_cur,year,temp_data,ToptA_t,lat)
        switch1 = switch_sexual(t_cur+1,year,temp_data,ToptA_t,lat)
        switch2 = switch_sexual(t_cur+2,year,temp_data,ToptA_t,lat)
        switch3 = switch_sexual(t_cur+3,year,temp_data,ToptA_t,lat)
        switch4 = switch_sexual(t_cur+4,year,temp_data,ToptA_t,lat)
        if switch == 1 and switch1 == 1 and switch2 == 1 and switch3 == 1 and switch4 == 1:
            date = t_cur
            print(f"year{year}: day {t_cur} - enter sexual phase")
            for ind in np.arange(i+1,365/dt):
                ind = int(ind)
                A[ind] = 0; L[ind] = 0; ToptAs[ind] = ToptA_end; ToptLs[ind] = ToptL_end; vG_A[ind] = vG_A_end[year]; vG_L[ind] = vG_L_end[year]
            break

    if num_change_A == 0:
        t_addA = np.nan; t_addL = np.nan
    elif num_change_A == 1 and num_change_L == 0:
        t_addL = np.nan

    outputs = [A,L,ToptAs,ToptLs,vG_A,vG_L,w_A,w_L,A_end,L_end,ToptA_end,ToptL_end]
    toc = time.perf_counter()
    print(f'total time for year{year}: {toc-tic} sec')
    return outputs

def get_output(fn,loc_dir_list):
    #input lon, lat, temp_data
    loc = fn.split(".csv", 1)[0]
    lon,lat = loc.split("_", 1)
    lon = float(lon)
    lat = float(lat)
    # if task == 'SK_SA':
    #     temp_data = dailytemp(fn,seas,trend)
    # else:
    #    temp_data = get_temps(lon,lat)
    temp_data = dailytemp(fn,seas,trend)

    #output filename
    fn = f'{lon}_{lat}_ALdata.csv'
    print(f'lon={lon} lat={lat}')
    out_file = os.path.join(loc_dir_list[0], fn)
    if os.path.exists(out_file):
        print(f'{lon}_{lat}_ALdata.csv exits!!!')
        return False

    if test_year == 1:
        A_add = 1000000
        L_add = 5000
    elif fn_id == 0 and test_year == 51:
        A_add = 32150406.264325
        L_add = 295625.451081619
    elif fn_id == 0 and test_year == 101:
        A_add = 32471112.3965455
        L_add = 273914.31867115
    elif fn_id == 1 and test_year == 31:
        A_add = 36129222.9697719
        L_add = 34914.2730379737
    if model == 'C+NP+NG'or model == 'NC+NP+NG':
        L_add = 0
    else:
        L_add = L_add
    Temp_max = np.max(temp_data)
    Temp_min = np.min(temp_data)


    if task == 'SA':
        Topt = pd.read_csv(f'../../inputs/smooth2000/Topt_equ_ov0.2_vGA0.3_vGL0.3_vEA0.7_vEL0.7_{code_version}/{lon}_{lat}.csv')
    elif task == '9points' or task == 's0t1':   
        Topt = pd.read_csv(f'../../inputs/smooth2000/Topt_equ_ov0.2_vGA0.3_vGL0.3_vEA0.7_vEL0.7_{code_version}/{lon}_{lat}.csv') 
    elif task == 's1t0':
        Topt = pd.read_csv(f'../../inputs/smooth2000_{task}/Topt_equ_ov0.2_vGA0.3_vGL0.3_vEA0.7_vEL0.7_{code_version}/{lon}_{lat}.csv')
    elif task == 'SK_SA':
        Topt = pd.read_csv(f'../../inputs/smooth2000_{task}/Topt_equ_ov0.2_vGA0.3_vGL0.3_vEA0.7_vEL0.7_{code_version}/{lon}_{lat}_{seas}.csv')

    ToptA0 = Topt.ToptA[0]
    ToptL0 = Topt.ToptL[0]
    print(f'ToptA0 = {ToptA0}, ToptL0 = {ToptL0}')
    print(f'Temp_min = {Temp_min},Temp_max = {Temp_max}')

    Adens_p = [];Ldens_p = []
    ToptAss = [];ToptLss  = [];vG_As = []; vG_Ls = []
    def year_loop(year,A_add,L_add,ToptA_end,ToptL_end,ov_mor=ov_mor):
        outputs_p = Solve_euler_model(year,A_add,L_add,temp_data,lat,ToptA_end,ToptL_end,t_start = 0, t_end = 365,dt=0.5)
        Aden_p = outputs_p[0]; Lden_p = outputs_p[1]
        ToptAs = outputs_p[2]; ToptLs = outputs_p[3]
        vG_A = outputs_p[4]; vG_L = outputs_p[5]
        A_end = outputs_p[8]; L_end = outputs_p[9]
        ToptA_end = outputs_p[10]; ToptL_end = outputs_p[11]

        Adens_p.extend(Aden_p)
        Ldens_p.extend(Lden_p)
        ToptAss.extend(ToptAs)
        ToptLss.extend(ToptLs)
        vG_As.extend(vG_A)
        vG_Ls.extend(vG_L)
        A_surv = A_end*(1-ov_mor);L_surv = L_end*(1-ov_mor)
        print(A_surv,L_surv)
        year = year + 1
        return year,A_surv,L_surv,ToptA_end,ToptL_end
    
    global vmA
    global vmL
    global vE_A_sqrt
    global vE_L_sqrt
    global vG_A0
    global vG_L0
    global vE_A
    global vE_L
    global dA_sv
    global vG_A_end
    global vG_L_end


    year = 1; ToptA_initial = ToptA0; ToptL_initial = ToptL0; 
    while year <= num_years:
        if task == 'SA':
            vG_A0 = args.vG_A0; vG_L0 = args.vG_L0; vE_A = args.vE_A; vE_L = args.vE_L
        else:
            if model == 'C+P+G':
                vG_A0 = args.vG_A0; vG_L0 = args.vG_L0; vE_A = args.vE_A; vE_L = args.vE_L
            else:
                if year <= 20:
                    vG_A0 = 0.3; vG_L0 = 0.3; vE_A = 0.7; vE_L = 0.7
                else:
                    vG_A0 = args.vG_A0; vG_L0 = args.vG_L0; vE_A = args.vE_A; vE_L = args.vE_L
        # genetic parametes
        # can do sensitivity analysis on these parameters
        vmA = 0.001*vE_A*(1/10)
        vmL = 0.001*vE_L*(1/50)
        vE_A_sqrt = math.sqrt(vE_A)
        vE_L_sqrt = math.sqrt(vE_L)
        print(model)
        print(f'year{year}: vG_A{vG_A0}_vG_L{vG_L0}_vE_A{vE_A}_vE_L{vE_L}')

        year,A_add, L_add,ToptA_initial,ToptL_initial = year_loop(year,A_add,L_add,ToptA_initial,ToptL_initial)
        
        Adens_p= np.array(Adens_p)
        Ldens_p= np.array(Ldens_p)
        ToptAss= np.array(ToptAss)
        ToptLss= np.array(ToptLss)
        vG_As = np.array(vG_As)
        vG_Ls = np.array(vG_Ls)

        years = np.linspace(0,year-1,int((year-1)*365/dt))

        # plt.plot(years,Adens_p)
        # plt.title('Aphid')
        # fn = f'{lon}_{lat}_A_{year-1}.png'
        # out_file = os.path.join(loc_dir_list[1], fn)
        # plt.savefig(out_file,bbox_inches='tight')
        # plt.close()

        # plt.plot(years,Ldens_p)
        # plt.title('Ladybird')
        # fn = f'{lon}_{lat}_L_{year-1}.png'
        # out_file = os.path.join(loc_dir_list[2], fn)
        # plt.savefig(out_file,bbox_inches='tight')
        # plt.close()

        # plt.plot(years,ToptAss, label= 'ToptA')
        # plt.plot(years,ToptLss, label= 'ToptL')
        # plt.title('ToptA and ToptL')
        # plt.legend()
        # fn = f'{lon}_{lat}_ToptAL_{year-1}.png'
        # out_file = os.path.join(loc_dir_list[3], fn)
        # plt.savefig(out_file,bbox_inches='tight')
        # plt.close()

        # plt.plot(years,vG_As)
        # plt.title('Additive genetic variance for aphid')
        # fn = f'{lon}_{lat}_vG_A_{year-1}.png'
        # out_file = os.path.join(loc_dir_list[4], fn)
        # plt.savefig(out_file,bbox_inches='tight')
        # plt.close()

        # plt.plot(years,vG_Ls)
        # plt.title('Additive genetic variance for ladybird')
        # fn = f'{lon}_{lat}_vG_L_{year-1}.png'
        # out_file = os.path.join(loc_dir_list[5], fn)
        # plt.savefig(out_file,bbox_inches='tight')
        # plt.close()

        # save the dataframe of the important timings
        if year >= num_years:
            AL = np.vstack([Adens_p,Ldens_p,ToptAss,ToptLss,vG_As,vG_Ls])
            AL = pd.DataFrame(np.transpose(AL))
            AL.columns = ['Aden', 'Lden', 'ToptA', 'ToptL','vG_A','vG_L']
            fn = f'{lon}_{lat}_ALdata_{year-1}.csv'
            out_file = os.path.join(loc_dir_list[0], fn)
            if not os.path.isfile(out_file):
                AL.to_csv(out_file,header=True)

        Adens_p= Adens_p.tolist()
        Ldens_p= Ldens_p.tolist()
        ToptAss= ToptAss.tolist()
        ToptLss= ToptLss.tolist()
        vG_As = vG_As.tolist()
        vG_Ls = vG_Ls.tolist()


folder_list = ['data','A','L','ToptAL','vGA','vGL']
loc_dir_list = []
for folder in folder_list:
    if task == '9points' or task == 's0t1' or task == 's1t0':
        # vG_A0 = args.vG_A0; vG_L0 = args.vG_L0; vE_A = args.vE_A; vE_L = args.vE_L
        loc_dir = os.path.join('../../outputs', f'AdaptedTopt_{num_years}years_{climate}_{task}_{code_version}/{model}/ov{ov_mor}_vGA0.3_vGL0.3_vEA0.7_vEL0.7_steps{num_steps}',folder)
    elif task == 'SA':
        vG_A0 = args.vG_A0; vG_L0 = args.vG_L0; vE_A = args.vE_A; vE_L = args.vE_L
        loc_dir = os.path.join('../../outputs', f'AdaptedTopt_{num_years}years_{climate}_{task}_{code_version}/{model}/ov{ov_mor}_vGA{vG_A0}_vGL{vG_L0}_vEA{vE_A}_vEL{vE_L}_steps{num_steps}',folder)
    elif task == 'SK_SA':
        vG_A0 = args.vG_A0; vG_L0 = args.vG_L0; vE_A = args.vE_A; vE_L = args.vE_L
        loc_dir = os.path.join('../../outputs', f'AdaptedTopt_{num_years}years_{climate}_{task}_{code_version}/{model}/ov{ov_mor}_vGA{vG_A0}_vGL{vG_L0}_vEA{vE_A}_vEL{vE_L}_steps{num_steps}/s{seas}_k{trend}',folder)
    loc_dir_list.append(loc_dir)
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir,exist_ok=True)

dA_sv = np.zeros(num_years+1)
vG_A_end = np.zeros(num_years+1)
vG_A_end[0] = 0.3
vG_L_end = np.zeros(num_years+1)
vG_L_end[0] = 0.3

if task == '9points':
    fn_list = ['-84.0_32.0.csv','-84.0_38.0.csv','-84.0_44.0.csv','-94.0_32.0.csv','-94.0_38.0.csv','-94.0_44.0.csv','-104.0_32.0.csv','-104.0_38.0.csv','-104.0_44.0.csv']
elif task == 'SK_SA':
    fn_list = ['-94.0_32.0.csv','-94.0_38.0.csv','-94.0_44.0.csv']
else:
    fn_list = ['-94.0_32.0.csv','-94.0_44.0.csv']
get_output(fn_list[fn_id],loc_dir_list)