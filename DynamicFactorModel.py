# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:00:39 2020

@author: Hogan
"""
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
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
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from DiscreteKalmanFilter import *
from Functions import *

def DFM(observation, n_factors, n_shocks):
    # model: y_it = miu_i + lambda_i*F_t + e_it
    # observables: columns are times series variables in interests, rows are time points
    # n_factors: number of common factors, can't be larger than number of input variables
    if len(observation.columns)<=n_factors:
        return print('Error: number of common factors exceeds limit')
    # n_time: number of time periods (observations)
    n_time = len(observation.index)
    # function returns two elements: common factors in the format of that of observables, transform matrix n_factors*n_observables

        
    "pca"
    x = np.mat(observation-observation.mean())
    z = np.mat((observation-observation.mean())/observation.std())

    D, V, S = calculate_pca(observation, n_factors)
    Psi = np.mat(np.diag(np.diag(S - V.dot(D).dot(V.T))))
    factors = V.T.dot(z.T).T
    CommonFactors = pd.DataFrame(data=factors, index=observation.index, columns=['Factor' + str(i+1) for i in range(n_factors)])
    
    "Factors loadings"
    Lambda = calculate_factor_loadings(observation, CommonFactors)
    
    "VAR"
    # model: F_t = A*F_{t-1} + B*u_t
    # calculate matrix A and B
    A = calculate_prediction_matrix(CommonFactors)
    B, Sigma = calculate_shock_matrix(CommonFactors, A, n_shocks)
    
    return DFMResultsWrapper(common_factors=CommonFactors,Lambda=Lambda, A=A, B=B, idiosyncratic_covariance=Psi, prediction_covariance=Sigma, obs_mean=observation.mean())

class DFMResultsWrapper():
    def __init__(self, common_factors, Lambda, A, B,  idiosyncratic_covariance, prediction_covariance, obs_mean):
        self.common_factors = common_factors
        self.Lambda = Lambda
        self.A = A
        self.B = B
        self.idiosyncratic_covariance = idiosyncratic_covariance
        self.prediction_covariance = prediction_covariance
        self.obs_mean = obs_mean

def DFM_EMalgo(observation, n_factors, n_shocks, n_iter, error='False'):
    dfm = DFM(observation, n_factors, n_shocks)
    
    if error=='True':
        error = pd.DataFrame(data=rand_Matrix(len(observation.index), n_shocks),columns=['shock'+str(i+1) for i in range(n_shocks)],index=observation.index)
    else:
        error = pd.DataFrame(data=np.zeros(shape=(len(observation.index), n_shocks)),columns=['shock'+str(i+1) for i in range(n_shocks)],index=observation.index)

    kf = KalmanFilter(Z=observation-observation.mean(), U=error, A=dfm.A, B=dfm.B, H=dfm.Lambda, state_names=dfm.common_factors.columns, x0=dfm.common_factors.iloc[0], P0=calculate_covariance(dfm.common_factors), Q=dfm.prediction_covariance, R=dfm.idiosyncratic_covariance)
    
    fis = FIS(kf)
    
    for i in range(n_iter):
        em = EMstep(fis,n_shocks)
        start = em.Lambda.I.dot((em.z-em.z.mean()).T).T
        kf = KalmanFilter(Z=observation-observation.mean(), U=error, A=em.A, B=em.B, H=em.Lambda, state_names=dfm.common_factors.columns, x0=start[0], P0=calculate_covariance(em.x_sm), Q=em.Q, R=em.R)
        fis = FIS(kf)
    
    return DFMEMResultsWrapper(A=em.A, B=em.B, Q=em.Q, R=em.R, Lambda=em.Lambda, x=kf.x, x_sm=em.x_sm, z=kf.z)

class DFMEMResultsWrapper():
    def __init__(self, A, B, Q, R, Lambda, x, x_sm, z):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Lambda = Lambda
        self.x = x
        self.x_sm = x_sm
        self.z = z
    
def RevserseTranslate(Factors, miu, Lambda, names):
    observation=pd.DataFrame(data=Lambda.dot(Factors.T).T, columns=names,index=Factors.index)
    observation = observation+miu
    return observation