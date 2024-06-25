#%%
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import minimize
from scipy.optimize import minimize, Bounds
from scipy.stats import lognorm
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW

#%%

# Estimate all elements to implement the Disaggregation model
# INPUT:
# 1) Daily_Ts: the daily time series 
# 2) n_par: »n» parameter value

# OUTPUT:
# 1) plot_Daily_Ts: plot of the first year of the Daily Ts
# 2) plot_ann_max: plot of the annual maxima 
# 3) a_RP: matrix of the «a» coefficients estimated using Return Periods (from 2 to 500) 
#         according to Parametric and Empirical approaches
# 4) gev_pars: vector of the GEV parameters estimated using MLE
# 5) Mean_Intesity: the Mean Annual Rainfall Intensity

def res_for_MF_Dissagr(Daily_Ts, n_par=0.344):
    Daily_Ts.columns = ['Time', 'Value']
    
    # 1) Handle NA values
    Daily_Ts['Value'] = Daily_Ts['Value'].interpolate()
    Daily_Ts['Year'] = pd.DatetimeIndex(Daily_Ts['Time']).year.astype('category')
    
    # 2) Select the Maximum Annual Values, transform them into «a» coefficients
    Max = Daily_Ts.groupby('Year')['Value'].max().reset_index()
    
    a_par_values = ((Max['Value'] / 24) / (24 ** (n_par - 1))) * 1.12
    
    # 3) Fit the GEV distribution on the «a» values, using 'mle' method
    gev_pars = genextreme.fit(a_par_values)
    
    # 4) Estimate the Mean Annual Rainfall Intensity
    Mean_Intesity = Daily_Ts['Value'].sum() / (len(Daily_Ts) * 24)
    
    # Estimation of «a» parameters with respect to several RPs
    Return_periods = np.arange(2, 501)
    
    # Parametric approach
    a_RP_mle = [genextreme.ppf(1 - (1 / rp), *gev_pars) for rp in Return_periods]
    
    # Empirical approach
    Tb = Return_periods * 365
    appo2 = Daily_Ts['Value'].to_numpy()
    ecdf_function = ECDF(appo2)
    inverse_ecdf = interp1d(ecdf_function.y, ecdf_function.x, bounds_error=False, fill_value="extrapolate")
    a_RP_empir = [inverse_ecdf(1 - 1 / tb) for tb in Tb]
    
    a_RP_par = np.round(a_RP_mle, 2)
    a_RP_emp = np.round(a_RP_empir, 2)
    
    a_RP = pd.DataFrame([a_RP_par, a_RP_emp], columns=[f'RP{rp}' for rp in Return_periods])
    
    # Plotting
    plot_Daily_Ts = Daily_Ts.iloc[:365].plot(x='Time', y='Value', title='Daily Ts: First year', legend=False)
    plot_Daily_Ts.set_xlabel("Time")
    plot_Daily_Ts.set_ylabel("mm")
    plt.show()
    
    plot_ann_max = Max.plot(x='Year', y='Value', title='Plot of the Annual Maxima', legend=False)
    plot_ann_max.set_xlabel("Time")
    plot_ann_max.set_ylabel("mm")
    plt.show()
    
    res = {
        'n_par': n_par,
        'Daily_Ts_plot': plot_Daily_Ts,
        'ann_max_plot': plot_ann_max,
        'a_RP': a_RP,
        'gev_pars': gev_pars,
        'Mean_Intesity': Mean_Intesity,
        'Daily_Ts': Daily_Ts
    }
    return res

# Read the data
daily_data = pd.read_csv("C:/Users/Aldis/Documents/Master Data Science/NYU/Rainfall_Daily_Data.csv")
Daily_Ts = daily_data.iloc[:, [0, 9]]

# Convert character to a "Date" object
Daily_Ts['Time'] = pd.to_datetime(Daily_Ts['Time'])

# Call the function with the given data
res1 = res_for_MF_Dissagr(Daily_Ts, n_par=0.247)

# Print results
print("GEV Parameters:", res1['gev_pars'])
print("Mean Intensity:", res1['Mean_Intesity'])
print("a_RP:\n", res1['a_RP'])
#%%


# Numerical problem optimization
# INPUT:   
# 1) res1: the output of res_for_MF_Dissagr() function;
# 2) lower_bounds and upper_bounds: the ranges of parameter values;
# 3) Return periods: a vector with a length up to 10 values;
# 4) method = c('parametric','empirical'): it is the method used to estimate "a" coefficients (and h(mm));
# 5) Num_Iterations: the maximum number of iterations in the optimization problem
  
# OUTPUT:
# 1) Cb, Cln, Duration (3 parameters);
# 2) N_iterations: the number of iterations required for the objective function to reach the optimal value;
# 3) objfun_value: the optimal value achieved by the objective function;
# 4) comparison_plot: plot for comparing the results.


# Función auxiliar para calcular Kq
def Kq(q, alpha, c_beta, c_ln=None):
    if alpha < 0:
        kqc = c_beta * (q - 1) + c_ln * (q**2 - q)
    elif alpha > 2:
        kqc = np.log2(c_beta / (c_beta - q)) - q * np.log2(c_beta / (c_beta - 1))
        kqc[q >= c_beta] = np.inf
    else:
        kqc = (c_beta / (alpha - 1)) * (q * (q ** (alpha - 1) - 1))
    
    kqlim = q - 1
    valkq = np.abs(kqc) < np.abs(kqlim)
    kqc[~valkq] = kqlim[~valkq]
    
    if np.any(q == 1):
        qm1 = q < 1
        if np.any(qm1):
            if np.all(valkq[qm1]):
                kqc[q == 1] = 0
        else:
            kqc[q == 1] = 0
    return kqc

# Función auxiliar para calcular momentos
def momz(qmax, alpha, c_beta, c_ln=None):
    eta = 1
    qc = np.arange(qmax + 1)
    if c_ln is None:
        kqc = Kq(qc, alpha, c_beta)
    else:
        kqc = Kq(qc, alpha, c_beta, c_ln)
    
    ez = np.ones(qmax + 1)
    bcoef = [1, 1]
    
    for k in range(2, qmax + 1):
        ez1 = ez[2:k+1]
        ez2 = np.flip(ez1)
        bcoef = [1] + [bcoef[i] + bcoef[i+1] for i in range(len(bcoef) - 1)] + [1]
        bc = bcoef[1:-1]
        ezaux = ez1 * ez2 * bc * (2 ** (kqc[2:k+1] + np.flip(kqc[2:k+1])))
        if k > 2:
            eza = np.sum(ezaux)
        else:
            eza = np.sum([np.sum(x) for x in ezaux])
        ez[k + 1] = (2 ** -k) / (1 - 2 ** (kqc[k + 1] - k + 1)) * eza
    
    ez[np.isinf(ez)] = np.inf
    return ez

# Función para comparar idfs teóricas y empíricas
def Comparison_Emp_idf_vs_Theor_idf(Cb, Cln, D, Return_periods, h_mm, MeanIntesity, n_par, Num_Iterations):
    RP = [rp * 365 * 24 for rp in Return_periods]
    coeff_a = [(h_mm[ii] / 24) / (24 ** (n_par - 1)) for ii in range(len(Return_periods))]
    
    durations = np.array([1/3, 1, 2, 6, 24])
    erre = D / durations
    nr = len(erre)
    
    qstar = (1 - Cb) / Cln
    alpha = -1
    MomentOrder1 = round(qstar / 2)
    Ez = momz(MomentOrder1, alpha, Cb, Cln)
    rz = Ez[MomentOrder1] ** (1 / (Cb * (MomentOrder1 - 1) + Cln * (MomentOrder1**2 - MomentOrder1)))
    i_star = (erre * rz) ** (2 - Cb - Cln)
    ratio = ((1 - Cb) ** 2) / Cln
    T_star = durations * ((2 * np.pi * 2 * np.log(erre * rz) * ratio) ** 0.5) * (erre * rz) ** (ratio + Cb)
    
    Theor_idf = []
    for k in range(len(Return_periods)):
        argo = 1 - (((erre * rz) ** Cb) * D) / (erre * RP[k])
        i0 = ((erre * rz) ** (Cb - Cln)) * np.exp(((2 * Cln * np.log(erre * rz)) ** 0.5) * np.array([np.inf if arg <= 0 else np.log(arg) for arg in argo]))
        i0 = i0 * MeanIntesity
        istar = i_star * (RP[k] / T_star) ** (Cln / (1 - Cb))
        Theor_idf.append(np.where(RP[k] <= T_star, i0, istar))
    
    Empic_idf = [(coeff_a[ii] * (D / erre) ** (n_par - 1)) for ii in range(len(Return_periods))]
    
    Theor_idf_df = pd.DataFrame(np.array(Theor_idf).T, columns=[f'theor_idf_{rp}' for rp in Return_periods])
    Empic_idf_df = pd.DataFrame(np.array(Empic_idf).T, columns=[f'emp_idf_{rp}' for rp in Return_periods])
    
    return erre, Theor_idf_df, Empic_idf_df

# Definir la función objetivo
def obj_fun(par, Return_periods, h_mm, MeanIntesity, n_par, Num_Iterations):
    Cb, Cln, D = par
    erre, Theor_idf, Empic_idf = Comparison_Emp_idf_vs_Theor_idf(Cb, Cln, D, Return_periods, h_mm, MeanIntesity, n_par, Num_Iterations)
    diff = Theor_idf.to_numpy() - Empic_idf.to_numpy()
    return np.sum(diff**2)

# Problema de optimización
def NumOpt_problem(res1, lower_bounds, upper_bounds, Return_periods, method='parametric', Num_Iterations=50000):
    n_par = res1['n_par']
    Daily_Ts = res1['Daily_Ts']
    MeanIntesity = res1['Mean_Intesity']
    a_RP = res1['a_RP']
    
    if method == 'parametric':
        h_mm = [a_RP.iloc[0, rp - 1] * (24 ** (n_par - 1)) * 24 for rp in Return_periods]
    else:
        h_mm = [a_RP.iloc[1, rp - 1] * (24 ** (n_par - 1)) * 24 for rp in Return_periods]
    
    h_mm = np.round(h_mm, 2)
    
    bounds = Bounds(lower_bounds, upper_bounds)
    opt_result = minimize(obj_fun, x0=lower_bounds, args=(Return_periods, h_mm, MeanIntesity, n_par, Num_Iterations), bounds=bounds, options={'maxiter': Num_Iterations, 'disp': True})
    
    Cb, Cln, D = opt_result.x
    erre, Theor_idf, Empic_idf = Comparison_Emp_idf_vs_Theor_idf(Cb, Cln, D, Return_periods, h_mm, MeanIntesity, n_par, Num_Iterations)
    
    # Plotting
    log_r = np.log10(erre)
    log_theor_idf = np.log10(Theor_idf)
    log_emp_idf = np.log10(Empic_idf)
    
    for rp in Return_periods:
        plt.plot(log_r, log_theor_idf[f'theor_idf_{rp}'], label=f'Theor_idf RP{rp}')
        plt.plot(log_r, log_emp_idf[f'emp_idf_{rp}'], label=f'Emp_idf RP{rp}', linestyle='--')
    
    plt.xlabel('T (log scale)')
    plt.ylabel('i (log scale)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    
    res = {
        'Daily_Ts': Daily_Ts,
        'n_par': n_par,
        'Cb': round(Cb, 2),
        'Cln': round(Cln, 3),
        'Duration': round(D, 1),
        'N_iterations': round(opt_result.nit, 0),
        'objfun_value': round(opt_result.fun, 2),
        'comparison_plot': plt
    }
    return res

# Ejemplo de uso
lower_bounds = [0.1, 0.01, 24]
upper_bounds = [0.7, 0.1, 24000]
Return_periods = [2, 4, 6, 8, 10]
Num_Iterations = 50000

res2 = NumOpt_problem(res1, lower_bounds, upper_bounds, Return_periods, method='parametric', Num_Iterations=Num_Iterations)
print(res2)

#%%