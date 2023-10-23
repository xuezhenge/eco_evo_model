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
from sklearn import linear_model

parser = argparse.ArgumentParser()
parser.add_argument('--num_cores', type=int,
    default=7)
parser.add_argument('--num_years', type=int,
    default=5)
parser.add_argument('--climate', type=str,
    default='smooth')
parser.add_argument('--dt', type=float,
    default=0.5)
parser.add_argument('--fn_id', type=int,
    default=0)
parser.add_argument('--ov_mor', type=float,
    default=0.2)
parser.add_argument('--vG_A0', type=float,
    default=0.3, help='0,0.075,0.15,0.3,0.6,1.2')
parser.add_argument('--vG_L', type=float,
    default=0.3,help='0,0.075,0.15,0.3,0.6,1.2')
parser.add_argument('--vE_A', type=float,
    default=0.7, help='0,0.175,0.35,0.7,1.4,2.8')
parser.add_argument('--vE_L', type=float,
    default=0.7)
parser.add_argument('--equ_year', type=int,
    default=2000)
parser.add_argument('--task', type=str,
    default='SK_SA')
parser.add_argument('--num_steps', type=int,
    default=150)
parser.add_argument('--seas', type=float,
    default=0)
parser.add_argument('--trend', type=float,
    default=0)


code_version = 'v4.0'

args = parser.parse_args()
num_cores = args.num_cores
num_years = args.num_years
dt = args.dt
ov_mor = args.ov_mor
climate = args.climate
fn_id = args.fn_id
# find the Topt equillibrium in year xxx
equ_year = args.equ_year
task = args.task
num_steps = args.num_steps
seas = args.seas
trend = args.trend

# parameters
q1A = 1.5
K_A = 50000000

# parameters for predation rate
a = 0.000001
h = 0.01
mL = 1.0
q1L = 1.5

q2A = 1
q2L = 1
min_diff = -15
max_diff = 10
Topt0 = 25

# genetic parametes
# can do sensitivity analysis on these parameters
vG_A0 = args.vG_A0
vG_L = args.vG_L
vE_A = args.vE_A
vE_L = args.vE_L

vm = 0.001*vE_A
vE_A_sqrt = math.sqrt(vE_A)
vE_L_sqrt = math.sqrt(vE_L)

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

# temperature performance curve
def thorneley_france(m,Topt,q1,q2,T,deltaE_A):
    Topt = Topt+deltaE_A
    Tmin = Topt - 15
    Tmax = Topt + 10
    # Generic temperature-dependent function
    if (T>=(Tmin)) & (T<=(Tmax)):
        return m*(((T-(Tmin))**q1)*((Tmax-T)**q2))/(((Topt-Tmin)**q1)*((Tmax-Topt)**q2))
    else:
        return 0.0

# food dependent predation rate
def fdpr(a,h,A):
    return a/(1+a*h*A)

# save temperature data every year
def get_temps(lon,lat,num_years):
    fn = f'{lon}_{lat}.csv'
    temp_df = dailytemp(fn,seas)
    temp = np.array(temp_df.Temp)
    temp = np.tile(temp, num_years)
    temp_data = temp.reshape((num_years,365))
    return temp_data

def dailytemp(fn,seas):
    # without trend
    if fn == '-84.0_32.0.csv':
        Tmean0 = 15.973
    elif fn == '-84.0_38.0.csv':
        Tmean0 = 10.723
    elif fn == '-84.0_44.0.csv':
        Tmean0 = 6.477
    elif fn == '-94.0_32.0.csv':
        Tmean0 = 16.889
    elif fn == '-94.0_38.0.csv':
        Tmean0 = 11.252
    elif fn == '-94.0_44.0.csv':
        Tmean0 = 5.103
    elif fn == '-104.0_32.0.csv':
        Tmean0 = 13.318
    elif fn == '-104.0_38.0.csv':
        Tmean0 = 5.055
    elif fn == '-104.0_44.0.csv':
        Tmean0 = 4.153
    ts = np.arange(365)
    temps = []
    for t in ts:
        if equ_year == 2000:
            temp = -seas*math.cos(2*math.pi*t/365) + Tmean0
        elif equ_year == 2150:
            Tmean_2150 = Tmean0 + trend*1.5
            temp = -seas*math.cos(2*math.pi*t/365) + Tmean_2150
        temps.append(temp)
    temps = pd.DataFrame(np.array(temps),columns = ['Temp'])
    return temps


def get_daily_temp(t,year,temp_data):
    t = int(np.floor(t))
    if t == 365:
        Temp_t = temp_data[year-1,:][364]
    else:
        Temp_t = temp_data[year-1,:][t]
    return Temp_t

def interpolated_temp(t,year,temp_data):
    #integrate temperature at each dt
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
    Tmin = Topt - 20
    Tmax = Topt + 5
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
    # import pdb;pdb.set_trace()
    a1 = roots.get(a1)
    a2 = roots.get(a2)
    b1 = roots.get(b1)
    b2 = roots.get(b2)
    # print(f"eta: {time.time() - tic}")
    return a1,a2,b1,b2

def mor_rate(T,a1,a2,b1,b2, Topt,deltaE_A,v_min,v_max):
    # the function of temperature dependent mortality rate
    Topt = Topt + deltaE_A
    Tmin = Topt - 15
    Tmax = Topt + 10
    Topt1 = Topt - 5
    Topt2 = Topt + 5
    shift = Topt-25
    if T >= Tmin and T <= Topt1:
        rate = a1*(T - shift) + b1
    elif T > Topt1 and T < Topt2:
        rate = v_min
    elif T >= Topt2 and T <= Tmax:
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
#     TDCF = temp_dependent_conversion_factor(mL,Topt,q1L,q2L,Temp,0.58,1)
#     mA = mA_max*TDCF
#     return mA

def get_range(v):
    #https://homepage.divms.uiowa.edu/~mbognar/applets/normal.html
    if v == 0 or v == 0.0001:
        rangeV = 0.04
    elif v == 0.3:
        rangeV = 2
    elif v == 0.6:
        rangeV = 3
    elif v == 0.7:
        rangeV = 3
    elif v == 0.9:
        rangeV = 4
    elif v == 1.2:
        rangeV = 4
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
    vG_A = np.empty(n_t);vG_A.fill(vG_A0)
    w_A = np.zeros([n_t]);w_L = np.zeros([n_t])
    num_change_A = 0; num_change_L = 0

    for i in np.arange(1,n_t):
        t = ts[i-1] #previous time step
        t_cur = ts[i] #current time step
        # Temperature at current time step
        Temp_t_cur = interpolated_temp(t_cur,year,temp_data)

        # Shift and vG_A at previous time step
        ToptA_t = ToptAs[i-1]; ToptL_t = ToptLs[i-1];vG_A_t = vG_A[i-1]
        vG_A_sqrt = math.sqrt(vG_A_t)
        vG_L_sqrt = math.sqrt(vG_L)

        #mA and Qp
        mA = 0.6
        Qp = get_Qp(ToptL_t,Temp_t_cur)

        if Temp_t_cur < ToptA_t - 15 and num_change_A == 0:
            continue
        else:
            def get_AL(t,A_t,L_t):
                fdpr_effect = fdpr(a,h,A_t)
                carrying_effect = (1-A_t/K_A)
                # numerical integration for aphid and ladybird over environmental effect
                #for aphid
                def p_integA(ToptA_G,ToptA_t,ToptL_t,A_t,L_t,Temp_t_cur,beta):
                    if vE_A != 0:
                        rhoGA = norm.pdf(ToptA_G, loc = ToptA_t, scale = vG_A_sqrt)
                        expr = lambda deltaE_A: (thorneley_france(mA,ToptA_G,q1A,q2A,Temp_t_cur,deltaE_A)*carrying_effect - morAL(Temp_t_cur,ToptA_G,deltaE_A,idx = 'aphid'))*norm.pdf(deltaE_A, loc = 0, scale = vE_A_sqrt)
                        integrate_E = integrate.quad(expr, -rangeVEA, rangeVEA)[0]
                        numer_rhoA = rhoGA + rhoGA*integrate_E*dt - rhoGA*beta*dt
                    # else:
                    #     integrate_E = (thorneley_france(mA,ToptA_G,q1A,q2A,Temp_t_cur,0)*carrying_effect - morAL(Temp_t_cur,ToptA_G,0,idx = 'aphid'))*norm.pdf(ToptA_G, loc = ToptA_t, scale = vG_A_sqrt)
                    return numer_rhoA

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
                rangeVGL = get_range(vG_L)
                rangeVEL = get_range(vE_L)

                stepA = rangeVGA*2/num_steps
                stepL = rangeVGL*2/num_steps

                ToptA_Gs = np.arange(ToptA_t - rangeVGA,ToptA_t + rangeVGA +stepA,stepA)
                ToptL_Gs = np.arange(ToptL_t - rangeVGL,ToptL_t + rangeVGL +stepL,stepL)
                # for aphid
                if vG_A0 == 0:
                    new_ToptA = ToptA_t
                    new_vG_A = vG_A0
                    w_A[i] = thorneley_france(mA,ToptA_t,q1A,q2A,Temp_t_cur,0)*carrying_effect - fdpr_effect*thorneley_france(mL,ToptL_t,q1L,q2L,Temp_t_cur,0)*L_t - morAL(Temp_t_cur,ToptA_t,0,idx = 'aphid')
                else:
                    if vG_L == 0:
                        beta = fdpr_effect*thorneley_france(mL,ToptL_t,q1L,q2L,Temp_t_cur,0)*L_t
                        FA = thorneley_france(mA,ToptA_t,q1A,q2A,Temp_t_cur,0)*carrying_effect - morAL(Temp_t_cur,ToptA_t,0,idx = 'aphid')
                    else:
                        expr_beta = lambda ToptL_G,deltaE_L: fdpr_effect*thorneley_france(mL,ToptL_G,q1L,q2L,Temp_t_cur,deltaE_L)*L_t*norm.pdf(ToptL_G, loc = ToptL_t, scale = vG_L_sqrt)*norm.pdf(deltaE_L, loc = 0, scale = vE_L_sqrt)
                        beta = integrate.dblquad(expr_beta, -rangeVEL, rangeVEL,ToptL_t-rangeVGL, ToptL_t + rangeVGL)[0]
                        # expr_FA = lambda ToptA_G,deltaE_A: (thorneley_france(mA,ToptA_t,q1A,q2A,Temp_t_cur,0)*carrying_effect - morAL(Temp_t_cur,ToptA_t,0,idx = 'aphid'))*norm.pdf(ToptA_G, loc = ToptA_t, scale = vG_A_sqrt)*norm.pdf(deltaE_A, loc = 0, scale = vE_A_sqrt)
                        # FA = integrate.dblquad(expr_FA, -rangeVEA, rangeVEA,ToptA_t-rangeVGA, ToptA_t + rangeVGA)[0]
                    # numerical integration for aphid over genetic and environmental effect
                    resA = Parallel(n_jobs=num_cores)(delayed(p_integA)(item,ToptA_t,ToptL_t,A_t,L_t,Temp_t_cur,beta) for item in ToptA_Gs)
                    numerA = sum([ToptA_Gs[i] * resA[i]*stepA for i in range(len(resA))])
                    denomA = sum([resA[i]*stepA for i in range(len(resA))])

                    # # new mean of the shiftA after selection
                    new_ToptA = numerA/denomA
                    # new additive 
                    new_vG_A = sum([(ToptA_Gs[i]- new_ToptA)**2 * resA[i]*stepA/denomA for i in range(len(resA))])
                    w_A[i] = denomA - 1
                # for ladybird
                if vG_L == 0:
                    new_ToptL = ToptL_t
                    w_L[i] = fdpr_effect*thorneley_france(mL,ToptL_t,q1L,q2L,Temp_t_cur,0)*A_t/Qp - morAL(Temp_t_cur,ToptL_t,0,idx = 'ladybird')
                else:
                    # expr_wL = lambda ToptL_G,deltaE_L: (fdpr_effect*thorneley_france(mL,ToptL_t,q1L,q2L,Temp_t_cur,0)*A_t/Qp - morAL(Temp_t_cur,ToptL_t,0,idx = 'ladybird'))*norm.pdf(ToptL_G, loc = ToptL_t, scale = vG_L_sqrt)*norm.pdf(deltaE_L, loc = 0, scale = vE_L_sqrt)
                    # wL_bar = integrate.dblquad(expr_wL, -rangeVEL, rangeVEL,ToptL_t-rangeVGL, ToptL_t + rangeVGL)[0]
                    resL = Parallel(n_jobs=num_cores)(delayed(p_integL)(item,ToptA_t,ToptL_t,A_t,L_t,Temp_t_cur) for item in ToptL_Gs)
                    numerL = sum([ToptL_Gs[i] * resL[i]*stepL for i in range(len(resL))])
                    denomL = sum([resL[i]*stepL for i in range(len(resL))])
                    #denomL = 1 + wL_bar*dt
                    new_ToptL = numerL/denomL
                    w_L[i] = denomL - 1

                if num_change_A == 0:
                    ToptAs[i] = ToptA_t
                    vG_A[i] = vG_A_t
                else:
                    ToptAs[i] = new_ToptA
                    vG_A[i] = new_vG_A
                if num_change_L == 0:
                    ToptLs[i] = ToptL_t
                else:
                    ToptLs[i] = new_ToptL

                A[i] = A_t*denomA
                L[i] = L_t*denomL
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
                # add ladybird into the model (wL >0 constantly for 3 days)
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
        if switch == 1:
            date = t_cur
            print(f"year{year}: day {t_cur} - enter sexual phase")
            for ind in np.arange(i+1,365/dt):
                ind = int(ind)
                A[ind] = 0; L[ind] = 0; ToptAs[ind] = ToptA_end; ToptLs[ind] = ToptL_end; vG_A[ind] = vG_A0
            break
    if num_change_A == 0:
        t_addA = np.nan; t_addL = np.nan
    elif num_change_A == 1 and num_change_L == 0:
        t_addL = np.nan
    outputs = [A,L,ToptAs,ToptLs,vG_A,w_A,w_L,A_end,L_end,ToptA_end,ToptL_end]
    toc = time.perf_counter()
    print(f'total time for year{year}: {toc-tic} sec')
    return outputs

def find_equ(fn):
    tic = time.perf_counter()
    loc = fn.split(".csv", 1)[0]
    print(loc)
    lon,lat = loc.split("_", 1)
    lon = float(lon)
    lat = float(lat)

    out_file = os.path.join(loc_dir, f'{fn}')
    if os.path.exists(out_file):
        print(f'{fn} exits!!!')
        return False

    temp_data = get_temps(lon,lat,num_years)
    Temp_max = np.max(temp_data)
    Temp_min = np.min(temp_data)
    print(f'Temp_min = {Temp_min},Temp_max = {Temp_max}')
    def Topt_iteration(ToptA,ToptL,equilibriumA,equilibriumL,kA):
        A_add =  1000000 
        L_add =  5000
        Adens_p = [];Ldens_p = []
        ToptAss = [];ToptLss  = [];vG_As = []
        w_As = []; w_Ls = []
        def year_loop(year,A_add,L_add,ToptA_end,ToptL_end,ov_mor=ov_mor):
            outputs_p = Solve_euler_model(year,A_add,L_add,temp_data,lat,ToptA_end,ToptL_end,t_start = 0, t_end = 365,dt=0.5)
            Aden_p = outputs_p[0]; Lden_p = outputs_p[1]
            ToptAs = outputs_p[2]; ToptLs = outputs_p[3]; vG_A = outputs_p[4]
            A_end = outputs_p[7]; L_end = outputs_p[8]
            ToptA_end = outputs_p[9]; ToptL_end = outputs_p[10]
            Adens_p.extend(Aden_p)
            Ldens_p.extend(Lden_p)
            ToptAss.extend(ToptAs)
            ToptLss.extend(ToptLs)
            vG_As.extend(vG_A)
            A_surv = A_end*(1-ov_mor);L_surv = L_end*(1-ov_mor)
            print(A_surv,L_surv)
            year = year + 1
            return year,A_surv,L_surv,ToptA_end,ToptL_end
        #year 1 to year 30
        year = 1; ToptA_initial = ToptA; ToptL_initial = ToptL
        while year <= num_years:
            print(f'year = {year}')
            year,A_add, L_add,ToptA_initial,ToptL_initial= year_loop(year,A_add,L_add,ToptA_initial,ToptL_initial)
        ToptAss= np.array(ToptAss[730:]).reshape(730*(num_years-1),1)
        ToptLss= np.array(ToptLss[730:]).reshape(730*(num_years-1),1)
        x = np.arange(730*(num_years-1)).reshape(730*(num_years-1),1)
        modelA = linear_model.LinearRegression()
        modelL = linear_model.LinearRegression()
        modelA.fit(x,ToptAss)
        modelL.fit(x,ToptLss)
        kA = modelA.coef_.item()
        kL = modelL.coef_.item()
        kAs.append(kA)
        kLs.append(kL)
        print(kA,kL,ToptA,ToptL)
        
        #For aphid:
        def find_equ1(kA,kAs,equilibriumA,ToptA):
            if kA < 0:
                if equilibriumA == 0:
                    ToptA = ToptA - 1
                    equilibriumA = 0
                elif equilibriumA == 0.5:
                    if kL < abs(kLs[-2]):
                        ToptA = ToptA
                        equilibriumA = 1
                    else:
                        ToptA = ToptA - 0.5
                        equilibriumA = 1
            elif kA > 0:
                if equilibriumA == 0:
                    ToptA = ToptA + 0.5
                    equilibriumA = 0.5
                elif equilibriumA == 0.5:
                    if kA < abs(kAs[-3]): 
                        ToptA = ToptA
                        equilibriumA = 1
                    else:
                        ToptA = ToptA + 0.5
                        equilibriumA = 1
            return ToptA,equilibriumA

        if equilibriumA == 1 and equilibriumL == 1:
            ToptA = ToptA; ToptL = ToptL
        else:
            ToptA,equilibriumA = find_equ1(kA,kAs,equilibriumA,ToptA)
            ToptL,equilibriumL = find_equ1(kL,kLs,equilibriumL,ToptL)
        print(f'ToptA = {ToptA}, ToptL = {ToptL}, equilibriumA = {equilibriumA},equilibriumL = {equilibriumL}')
        return ToptA,ToptL,equilibriumA,equilibriumL,kA

    equilibriumA = 0
    equilibriumL = 0
    kA = 0
    kAs = []
    kLs = []
    # if task == 's1t0':
    #     Topt0 = pd.read_csv(f'../../inputs/{climate}{equ_year}_{task}/max_Topt/{lon}_{lat}.csv')
    #     Topt0 = Topt0.max_Topt[0]
    # elif task == 'SK_SA':
    #     Topt0 = Temp_max
    # else:
    #     Topt0 = pd.read_csv(f'../../inputs/{climate}{equ_year}/max_Topt/{lon}_{lat}.csv')
    #     Topt0 = Topt0.max_Topt[0]
    if seas > 4:
        Topt0 = np.floor(Temp_max)
    else:
        Topt0 = np.ceil(Temp_max)
    print(f'Topt = {Topt0}')
    ToptA0 = Topt0; ToptL0 = Topt0
    while equilibriumA != 1 or equilibriumL != 1:
        ToptA0,ToptL0,equilibriumA,equilibriumL,kA = Topt_iteration(ToptA0,ToptL0,equilibriumA,equilibriumL,kA)

    # because aphid reaches the equilibrim eariler then ladybird, the estimated ToptL is more accurrate than ladybird, so we will do several extral test to check the estimated ToptA is accurate or not
    if equilibriumA == 1 and equilibriumL==1:
        if kA < 0:
            ToptA0_down1,ToptL0,equilibriumA,equilibriumL,kA_down1 = Topt_iteration(ToptA0-0.5,ToptL0,equilibriumA,equilibriumL,kA)
            if abs(kA_down1) < abs(kA):
                ToptA0_down2,ToptL0,equilibriumA,equilibriumL,kA_down2 = Topt_iteration(ToptA0-1,ToptL0,equilibriumA,equilibriumL,kA)
                if abs(kA_down2) < abs(kA_down1):
                    ToptA0 = ToptA0_down2
                else:
                    ToptA0 = ToptA0_down1
            elif abs(kA_down1) > abs(kA):
                ToptA0 = ToptA0
        elif kA > 0:
            ToptA0_up1,ToptL0,equilibriumA,equilibriumL,kA_up1 = Topt_iteration(ToptA0+0.5,ToptL0,equilibriumA,equilibriumL,kA)
            if abs(kA_up1) < abs(kA):
                ToptA0 = ToptA0_up1
                ToptA0_up2,ToptL0,equilibriumA,equilibriumL,kA_up2 = Topt_iteration(ToptA0+1,ToptL0,equilibriumA,equilibriumL,kA)
                if abs(kA_up2) < abs(kA_up1):
                    ToptA0 = ToptA0_up2
                else:
                    ToptA0 = ToptA0_up1
            elif abs(kA_up1) > abs(kA):
                ToptA0 = ToptA0    

    Topts = [[ToptA0,ToptL0]]
    Topts = pd.DataFrame(Topts,columns = ['ToptA','ToptL'])
    if task == 's1t0':
        Topts.to_csv(f'../../inputs/{climate}{equ_year}_{task}/Topt_equ_ov{ov_mor}_vGA{vG_A0}_vGL{vG_L}_vEA{vE_A}_vEL{vE_L}_{code_version}/{fn}',index=False)
    elif task == 'SK_SA':
        Topts.to_csv(f'../../inputs/{climate}{equ_year}_{task}/Topt_equ_ov{ov_mor}_vGA{vG_A0}_vGL{vG_L}_vEA{vE_A}_vEL{vE_L}_{code_version}/{lon}_{lat}_{seas}.csv',index=False)
    else:
        Topts.to_csv(f'../../inputs/{climate}{equ_year}/Topt_equ_ov{ov_mor}_vGA{vG_A0}_vGL{vG_L}_vEA{vE_A}_vEL{vE_L}_{code_version}/{fn}',index=False)
    print('file saved!!!')
    toc = time.perf_counter()
    print(f'total time for finding equilibrium: {(toc-tic)/3600} hours')

if task == 's1t0' or task == 'SK_SA':
    loc_dir = f'../../inputs/{climate}{equ_year}_{task}/Topt_equ_ov{ov_mor}_vGA{vG_A0}_vGL{vG_L}_vEA{vE_A}_vEL{vE_L}_{code_version}'
else:
    loc_dir = f'../../inputs/{climate}{equ_year}/Topt_equ_ov{ov_mor}_vGA{vG_A0}_vGL{vG_L}_vEA{vE_A}_vEL{vE_L}_{code_version}'
if not os.path.exists(loc_dir):
    os.makedirs(loc_dir,exist_ok=True)

if task == '9points':
    fn_list = ['-84.0_32.0.csv','-84.0_38.0.csv','-84.0_44.0.csv','-94.0_32.0.csv','-94.0_38.0.csv','-94.0_44.0.csv','-104.0_32.0.csv','-104.0_38.0.csv','-104.0_44.0.csv']
elif task == 'SK_SA':
    fn_list = ['-94.0_32.0.csv','-94.0_38.0.csv','-94.0_44.0.csv']
else:
    fn_list = ['-94.0_32.0.csv','-94.0_44.0.csv']
find_equ(fn_list[fn_id])