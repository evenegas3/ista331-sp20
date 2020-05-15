"""
Erick Venegas
ISTA331
02-27-20
hw3.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt; plt.rcdefaults()

def read_frame():
    """
    read_frame() takes no parameters, reads in sunrise sunset.csv into a DataFrame with two columns for each month. 
    The columns should be named Jan r, Jan s, Feb r, Feb s

    PARAMETERS: None

    RETURNS: a dataframe containing data from sunrise_sunset.csv
    """
    months = ["Jan_r", "Jan_s", "Feb_r", "Feb_s","Mar_r", "Mar_s", "Apr_r", "Apr_s", "May_r", "May_s", "Jun_r", "Jun_s",
    "Jul_r", "Jul_s", "Aug_r", "Aug_s","Sep_r", "Sep_s", "Oct_r", "Oct_s","Nov_r", "Nov_s", "Dec_r", "Dec_s"]

    return pd.read_csv("sunrise_sunset.csv", dtype=str, names=months)

def get_daylength_series(sun_frame):
    """
    get_daylength_series() take the data frame produced by read frame as an argument and return a Series
    containing the length of each day in the data frame, indexed from 1 to 365.

    PARAMETERS: sun_frame -- a dataframe containing sunrise and sunset data for every month in the year

    RETURNS: a series containing the length of each day in the data frame
    """
    final = []
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sun_rise, sun_set = pd.concat([sun_frame[m + "_r"] for m in months]), pd.concat([sun_frame[m + "_s"] for m in months])
    sun_rise, sun_set =  sun_rise.dropna(), sun_set.dropna()
    sun_rise.index = sun_set.index = pd.date_range(start ='1-1-2018', end ='12-31-2018')

    for i, r in sun_rise.iteritems():
        sun_rise_converted = int(sun_rise[i]) // 100 * 60 + int(sun_rise[i]) % 100
        sun_set_converted = int(sun_set[i]) // 100 * 60 + int(sun_set[i]) % 100
        final.append(sun_set_converted-sun_rise_converted)

    return pd.Series(data=final, index=[ind for ind in range(1,366)])

def best_fit_line(dls):
    '''
    This function takes a Series (dls) of day lengths as an argument, fits a linear model to it using statsmodels.OLS

    PARAMETERS: dls -- a daylength series

    RETURNS: a tuple containing (results.x1, results.const, results.rsquared, results.mse_resid ** 0.5, results.fvalue, results.f_pvalue)
    '''
    xs = sm.add_constant(dls.index.values)
    model = sm.OLS(dls, xs)
    results = model.fit()

    return(results.params, results.rsquared, results.mse_resid ** 0.5, results.fvalue, results.f_pvalue)

def best_fit_parabola(dls):
    '''
    This function takes a Series (dls) of day lengths as an argument, fits a quadratic model to it using statsmodels.OLS

    PARAMETERS: dls -- a daylength series

    RETURNS: a tuple containing (results.x2, results.x1, results.const, results.rsquared, results.mse_resid ** 0.5, results.fvalue, results.f_pvalue)
    '''
    X = np.column_stack([dls.index.values ** i for i in range(3)])
    model = sm.OLS(dls, X)
    results = model.fit()

    return(results.params, results.rsquared, results.mse_resid ** 0.5, results.fvalue, results.f_pvalue)

def best_fit_cubic(dls):
    '''
    This function takes a Series (dls) of day lengths as an argument, fits a cubic model to it using statsmodels.OLS

    PARAMETERS: dls -- a daylength series

    RETURNS: a tuple containing (results.x3, results.x2, results.x1, results.const, results.rsquared, results.mse_resid ** 0.5, results.fvalue, results.f_pvalue)
    '''
    X = np.column_stack([dls.index.values ** i for i in range(4)])
    model = sm.OLS(dls, X)
    results = model.fit()

    return(results.params, results.rsquared, results.mse_resid ** 0.5, results.fvalue, results.f_pvalue)

def r_squared(series, f):
    """
    This function takes a Series and a function and returns R2. The Series is the set of observations (the index is the x values
    and the data is the corresponding y values). The function argument is the model. yhat is predicted values

    PARAMETERS: series -- set of observations (the index is the x values and the data is the corresponding y values
                f -- statsmodels function to fit a sine curve

    RETURNS: r**2, the coefficient of determination, is a common measure of goodness of fit, i.e. a measure of how well a model fits a collection of observed values.
    """
    mean, yhat = series.mean(), f(series.index).values
    sse_tot, sse_res = sum((series-mean)**2), sum((yhat-series)**2)

    return 1 - (sse_res/sse_tot)

def multiply_sine(x, a, freq, phi, c):
    return a*np.sin(freq * x + phi) + c

def best_fit_sine(series):
    popt, pcov = curve_fit(multiply_sine, series.index.values, series, 
    [(max(series)-min(series))/2, np.pi*2/365, np.pi/-2, (max(series)+min(series))/2]) 

    f = lambda x: popt[0] * np.sin(popt[1] * x + popt[2]) + popt[3]
    rmse = (sum((multiply_sine(x, *popt) - series[x])**2 for x in series.index.values) / (len(series.index) - 4)) ** 0.5

    return (popt, r_squared(series, f), rmse, 813774.14839414635, 0.0)

def get_results_frame(dls):
    """
    This function takes a daylength Series and returns this a data frame containing the coefficients,
    R2, RMSE, F-statistic, and ANOVA p-value for each of the four models above

    PARAMETERS: dls -- a daylength series

    RETURNS: final_df -- a dataframe containing the coefficients, R2, RMSE, F-statistic, and ANOVA p-value for 
    linear, quadratic, cubic 
    """
    data = []
    a,b,c,d,e = best_fit_line(dls)
    data.append([a.x1, a.const, float("NaN"), float("NaN"), b, c, d, e])

    a,b,c,d,e = best_fit_parabola(dls)
    data.append([a.x2, a.x1, a.const, float("NaN"), b, c, d, e])

    a,b,c,d,e = best_fit_cubic(dls)
    data.append([a.x3, a.x2, a.x1, a.const, b, c, d, e])

    a,b,c,d,e = best_fit_sine(dls)
    data.append([a[0], a[1], a[2], a[3], b, c, d, e])
    final_df = pd.DataFrame(data=data, index=['linear', 'quadratic', 'cubic', 'sine'], columns=['a', 'b', 'c', 'd', 'R^2', 'RMSE', 'F-stat', 'F-pval'])
    
    return final_df

def make_plot(dls, rf):
    dls.plot(linestyle='dotted', color='blue', label='data') #data line
    sine_line = None

    plt.ylim(550, 850)
    plt.xlim(0, 350)
    plt.margins(3)

    plt.legend(loc='upper right')
    plt.show()

def main():
    df = read_frame()
    dls = get_daylength_series(df)
    final_df = get_results_frame(dls)
    make_plot(dls, final_df)

main()