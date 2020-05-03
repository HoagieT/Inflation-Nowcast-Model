# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:01:28 2020

@author: Hogan
"""
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import *
import tkinter.filedialog
from datetime import timedelta
import math 
import time
import calendar
from numba import jit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR, DynamicVAR
import scipy
import statsmodels.tsa.stattools as ts
import statsmodels.tsa as tsa
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from datetime import datetime,timedelta

def expand_data(data, step, freq='M'):
    dates = pd.date_range(data.index[0], periods=len(data.index)+step, freq=freq)
    data_expanded = pd.DataFrame(data=np.nan,index=dates,columns=data.columns)
    data_expanded.iloc[:len(data.index)] = data.values
    
    return data_expanded

def import_data(file_name, sheet_name, start=0, interpolation=False, encoding='gb18030'):
    Temp = pd.read_excel(file_name, sheetname = sheet_name, encoding = encoding)
    res = Temp.iloc[start:,1:]
    res.index = Temp.iloc[start:,0]
    if interpolation==True:
        res = DataInterpolation(res, 0, len(res.index), 'cubic').dropna(axis=0,how='any')
    return res

def transform_data(data, method):
    transform=pd.DataFrame(data=np.nan,index=data.index,columns=data.columns)
    if method=='MoM':
        for i in range(5,len(data.index)):
            last_month = data[data.index<=(data.index[i]-timedelta(30))].iloc[-1]
            transform.iloc[i] = (data.iloc[i]-last_month)/last_month*100
    return transform

def plot_compare(history, forecast, title,fig_size=[24,16], line_width=3.0,font_size='xx-large'):
    plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['image.cmap'] = 'gray'
    
    plt.figure()
    plt.plot(history.index,history,color='r',label='observed', linewidth=line_width)
    plt.plot(forecast.index,forecast,color='k',label='predicted', linewidth=line_width)
    plt.legend()
    plt.xlabel('Date')
    plt.title(title, fontweight='bold', fontsize=font_size)
    plt.show()
    
    return
          


def DataInterpolation(data, start, end, method):
    # data must be a time series dataframe
    n_row = len(data.index)
    n_col = len(data.columns)
    res = np.mat(np.zeros(shape=(n_row,n_col)))
    
    for i in range(n_col):
        res[:,i] = np.mat(data.iloc[:,i]).T
        y=data.iloc[start:end,i]
        location = np.where(y.notnull())[0]
        upper_bound=max(location)
        lower_bound=min(location)
        f2 = interp1d(location, y[y.notnull()], kind=method)
        x = np.linspace(lower_bound, upper_bound, num=upper_bound-lower_bound, endpoint=False)
        res[lower_bound:upper_bound,i]=np.mat(f2(x)).T
    
    res = pd.DataFrame(res, index=data.index, columns=data.columns)
    
    return res

def rand_Matrix(n_row, n_col):
    randArr = np.random.randn(n_row, n_col)
    randMat = np.mat(randArr)
    return randMat


def calculate_factor_loadings(observables, factors):
    # Both dataframes should have the same time stamp
    n_time = len(observables.index)
    x = np.mat(observables-observables.mean())
    F=np.mat(factors)
    temp = F[0].T.dot(F[0])
    for i in range(1,n_time):
        temp = temp + F[i].T.dot(F[i])
    
    Lambda = x[0].T.dot(F[0]).dot(temp.I)
    for i in range(1,n_time):
        Lambda = Lambda + x[i].T.dot(F[i]).dot(temp.I)
        
    return Lambda

def calculate_prediction_matrix(factors):
    n_time = len(factors.index)
    F=np.mat(factors)
    
    temp = F[0].T.dot(F[0])
    for i in range(2,n_time):
        temp = temp + F[i-1].T.dot(F[i-1])
    
    A = F[1].T.dot(F[0]).dot(temp.I)
    for i in range(2,n_time):
        A = A + F[i].T.dot(F[i-1]).dot(temp.I)
    
    return A

def calculate_shock_matrix(factors, prediction_matrix, n_shocks):
    n_time = len(factors.index)
    F = np.mat(factors)
    A = prediction_matrix
    
    temp = F[0].T.dot(F[0])
    for i in range(2,n_time):
        temp = temp + F[i-1].T.dot(F[i-1])
    
    term1 = F[1].T.dot(F[1])
    for i in range(2,n_time):
        term1 = term1 + F[i].T.dot(F[i])
    term1 = term1/(n_time-1)
    term2 = A.dot(temp/(n_time-1)).dot(A.T)
    Sigma = term1 - term2
    
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    sorted_indices = np.argsort(eigenvalues)
    evalues = eigenvalues[sorted_indices[:-n_shocks-1:-1]]
    M = eigenvectors[:,sorted_indices[:-n_shocks-1:-1]]
    B = M.dot(np.diag(pow(evalues,0.5)))
    
    return B, Sigma

def calculate_pca(observables, n_factors):
    # syntax: 
    n_time = len(observables.index)
    x = np.mat(observables-observables.mean())
    z = np.mat((observables-observables.mean())/observables.std())
    
    S = z[0].T.dot(z[0])
    for i in range(1,n_time):
        S = S + z[i].T.dot(z[i])
    
    eigenvalues, eigenvectors = np.linalg.eig(S)
    sorted_indices = np.argsort(eigenvalues)
    evalues = eigenvalues[sorted_indices[:-n_factors-1:-1]]
    V = np.mat(eigenvectors[:,sorted_indices[:-n_factors-1:-1]])
    D = np.diag(evalues)
    
    return D, V, S
    
def calculate_covariance(factors):
    n_time = len(factors.index)
    F = np.mat(factors)
    temp = [factors.iloc[:,i] for i in range(len(factors.columns))]
    return np.cov(temp) 