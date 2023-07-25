# data processing packages
import numpy as np
import pandas as pd

# stat packages
import scipy
from scipy import stats
from scipy.optimize import curve_fit
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats import *

# gaussian process
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# viz packages
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib import colors
import seaborn as sns

# other utility packages
import json
import geopy
import ast
import datetime
import calendar
from dateutil.relativedelta import relativedelta

# import customized packages/functions
from .preprocessing_and_utilities import *


# the major park types that attracted more than 90% of the total visits
major_park_types = ['Neighborhood Park', 'Community Park', 'Flagship Park', 'Jointly Operated Playground',
                    'Playground', 'Triangle/Plaza', 'Nature Area', 'Recreation Field/Courts']


### person correlation between monthly park visits by park type and monthly mean temperature

def prepare_data(park_visits_data, monthly_temp_data, value_cols):

    park_visits_monthly_total_pivot = park_visits_data.pivot(index=['date', 'year', 'month'], columns=['park_type'], values=value_cols).reset_index()
    # monthly_temp_data.rename(columns={'DATE':'date'}, inplace=True)
    data_for_test = pd.merge(park_visits_monthly_total_pivot, monthly_temp_data, left_on='date', right_on='DATE')
    data_for_test = data_for_test[data_for_test['date', ''] < datetime.datetime(2020,3,1)]    # only use park visits data before the pandemic

    return data_for_test


def pearson_correlation(data, value_col):

    columns = ['n_sample', 'r_value', 'p_value']
    pearson_results = pd.DataFrame(columns=columns)

    for park_type in major_park_types:
        
        pearson_results.loc[park_type, 'n_sample'] = data[value_col, park_type].shape[0]
        pearson_results.loc[park_type, 'r_value'] = stats.pearsonr(data[value_col, park_type], data['TAVG'])[0]
        pearson_results.loc[park_type, 'p_value'] = stats.pearsonr(data[value_col, park_type], data['TAVG'])[1]
            
    return pearson_results


### t test for temp difference in each month

def prepare_daily_temp_data(daily_temp_data):

    daily_temp_data['Year'] = daily_temp_data['DATE'].dt.year
    daily_temp_data['Month'] = daily_temp_data['DATE'].dt.month
    daily_temp_data['Date'] = daily_temp_data['DATE'].dt.strftime("%m-%d")

    daily_temp_data_pivot = daily_temp_data.pivot(index=['Date', 'Month'], columns=['Year'], values=['TAVG']).reset_index()

    return daily_temp_data_pivot


def t_test(data):

    columns = ['month', 'n_sample', 't_stat', 'p_value']
    ttest_results = pd.DataFrame(columns=columns)

    for month in range(1, 13):
        
        data_month_sub = data[data['Month'] == month].dropna()
        ttest_results.loc[month, 'month'] = month
        ttest_results.loc[month, 'n_sample'] = data_month_sub.shape[0]
        
        ttest_result = scipy.stats.ttest_rel(data_month_sub['TAVG', 2020],
                                             data_month_sub['TAVG', 2019])

        ttest_results.loc[month, 't_stat'] = ttest_result[0]
        ttest_results.loc[month, 'p_value'] = ttest_result[1]
            
    return ttest_results


### fit a model between park visits and temperature

## least square fit

# linear fit
def func_linear(x, a, b):
    return a * x + b

# exponential fit
def func_exp(x, a, b, c):
    return a * np.exp(-b*x) + c

# 2-degree polynomial fit
def func_poly2(x, a, b, c):
    return a * x**2 + b * x + c

# 3-degree polynomial fit
def func_poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def fit_curve(func, X,y, print_label=False):
    
    # fit the curve
    popt, pcov = curve_fit(func, X, y, maxfev=1000000)

    # calculate the r2 value
    residuals = y - func(X, *popt)
    ss_res = np.sum(residuals**2)

    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    if print_label:
        print("$R^2$" + print_label, r_squared)
    
    return popt, pcov, r_squared


def plot_least_square_fit_result(X, y, popt_1, popt_2, popt_3, popt_4):

    fig, ax = plt.subplots(figsize=(12,8))

    ax.plot(X, y, 'o', label='data')
    ax.plot(X, func_linear(X, *popt_1), '-', lw=2, label='linear fit: $%5.1f \cdot x + %5.1f$' % tuple(popt_1))
    ax.plot(X, func_exp(X, *popt_2), '-', lw=2, label='exp fit: $%5.1f \cdot \exp^{-%5.2f \cdot x}  + %5.1f$' % tuple(popt_2))
    ax.plot(X, func_poly2(X, *popt_3), '-', lw=2, label='2-degree poly fit: $%5.1f \cdot x^2 + %5.1f \cdot x + %5.1f$' % tuple(popt_3))
    ax.plot(X, func_poly3(X, *popt_4), '-', lw=2, label='3-degree poly fit: $%5.1f \cdot x^3 + %5.1f \cdot x^2 + %5.1f \cdot x + %5.1f$' % tuple(popt_4))

    ax.set_xlabel("Temperature $(°C)$")
    ax.set_ylabel("Number of Visits")

    ax.legend(fontsize=15);


## gaussian process

def gaussian_process_regression(X, y):
    
    # generate x for prediction
    x = np.atleast_2d(np.linspace(-5, 30, 1000)).T

    # initialize a Gaussian Process Regressor
    kernel = RBF(1.0, (1e-5, 1e5)) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)

    # make a pipeline and fit it
    pipe = make_pipeline(preprocessing.StandardScaler(), gp)
    pipe.fit(X, y)

    # make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = pipe.predict(x, return_std=True)

    return x, y_pred, sigma, pipe


def plot_gaussian_proccess_result(X, y, x, y_pred, sigma):

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(X, y, 'r.', markersize=10, label='Observations')
    ax.plot(x, y_pred, 'b-', label='Prediction')
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')

    ax.set_xlabel('Temperature $(°C)$')
    ax.set_ylabel('Number of Visits')
    ax.legend(loc='upper left');


def plot_all_model_results(X, y, x, y_pred, sigma, r2_gp,
                           popt_1, popt_2, popt_3, popt_4,
                           r2_1, r2_2, r2_3, r2_4):

    fig, ax = plt.subplots(figsize=(12,6))

    cmap = plt.cm.get_cmap('tab10', 10)

    # plot original data points
    ax.plot(X, y, 'o', c='r', label='Data')

    # plot OLS regression results
    ax.plot(X, func_linear(X, *popt_1), '-', lw=2, c=cmap.colors[0], label='Linear Fit - $R^2:${:.2f}'.format(r2_1))
    ax.plot(X, func_exp(X, *popt_2), '-', lw=2, c=cmap.colors[1], label='Exponential Fit - $R^2:${:.2f}'.format(r2_2))
    ax.plot(X, func_poly2(X, *popt_3), '-', lw=2, c=cmap.colors[2], label='2-Degree Polynomial Fit - $R^2:${:.2f}'.format(r2_3))
    ax.plot(X, func_poly3(X, *popt_4), '-', lw=2, c=cmap.colors[3], label='3-Degree Polynomial Fit - $R^2:${:.2f}'.format(r2_4))

    # plot Gaussian Process regression results
    ax.plot(x, y_pred, 'b--', label='Gaussian Process Model - $R^2:${:.2f}'.format(r2_gp))
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.4, fc='b', ec='None', label='Gaussian Process Model - 95% Confidence Interval')

    # ax settings
    ax.set_xlabel("Temperature $(°C)$")
    ax.set_ylabel("Number of Visits")

    ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True);


## adjust park visits by temperature

def adjust_park_visits_by_temp(park_visits_data, temp_data, func, popt, visit_col_2019, visit_col_2020, remove_extremes=False):

    park_visits_temp = pd.merge(park_visits_data, temp_data, left_on='date', right_on='DATE')

    # filter the data for each year
    park_visits_2019 = park_visits_temp[park_visits_temp['year'] == 2019]
    park_visits_2020 = park_visits_temp[park_visits_temp['year'] == 2020]

    # prepare to calculate the visits change rate
    park_visits_change = pd.merge(park_visits_2019, park_visits_2020, how='outer',
                                  on=['park_name', 'park_type', 'borough', 'month'],
                                  suffixes=['_2019', '_2020'])

    # drop redundant or duplicated columns
    park_visits_change.drop(columns=['date_2019','date_2020', 'DATE_2019', 'DATE_2020', 'year_2019', 'year_2020'], inplace=True)
    park_visits_change.dropna(subset=[visit_col_2019, visit_col_2020], inplace=True)

    # use the best model (3rd degree polynomial) to calculate the visits adjustment rate
    park_visits_change['visits_adj_rate'] = (func(park_visits_change['TAVG_2020'].to_numpy().reshape(-1,1), *popt) - func(park_visits_change['TAVG_2019'].to_numpy().reshape(-1,1), *popt)) / \
                                             func(park_visits_change['TAVG_2020'].to_numpy().reshape(-1,1), *popt)

    # calculate the adjusted base number of visits (in 2019)
    park_visits_change['visits_base_adjtd']  = park_visits_change[visit_col_2019] * (park_visits_change['visits_adj_rate'] + 1)

    # calculate the visits change rate
    park_visits_change['visit_change_rate'] = (park_visits_change[visit_col_2020] - park_visits_change['visits_base_adjtd']) / park_visits_change['visits_base_adjtd'] * 100
    park_visits_change_cat = park_visits_change.dropna(subset=['visit_change_rate'])
    # park_visits_change_cat = park_visits_change_cat[(park_visits_change_cat['park_type'].isin(major_park_types))]
    # #                                                 & (park_visits_change_cat['month'].isin(months))]

    # change certain columns' type to category
    park_visits_change_cat['park_type'] = park_visits_change_cat['park_type'].astype('category')
    park_visits_change_cat['month'] = park_visits_change_cat['month'].astype('category')
    park_visits_change_cat['borough'] = park_visits_change_cat['borough'].astype('category')

    # remove extreme values (use only the middle 95% of the data)
    if remove_extremes:
        park_visits_change_cat = remove_extreme_values(park_visits_change_cat, 'visit_change_rate')

    return park_visits_change_cat




