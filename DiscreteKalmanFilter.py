# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:00:22 2020

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
from Functions import *


class KalmanFilterResultsWrapper():
    def __init__(self, x_minus, x, z, Kalman_gain, P, P_minus, state_names, A):
        self.x_minus = x_minus
        self.x = x
        self.z = z
        self.Kalman_gain = Kalman_gain
        self.P = P
        self.state_names = state_names
        self.A = A
        self.P_minus = P_minus
        
def KalmanFilter(Z, U, A, B, H, state_names, x0, P0, Q, R):
    #x_t = A*x_{t-1} + B*u_t + Q
    #z_t = H*x_t + R
    # Q is process noise covariance
    # R is measurement noice covariance
    
    measurement_names = Z.columns
    
    timestamp = Z.index
    n_time = len(Z.index)
    n_state = len(state_names)
    z = np.mat(Z.as_matrix())
    u = np.mat(U.as_matrix())

    
    "out initialization"
    # predictions
    x = np.mat(np.zeros(shape=(n_time, n_state)))
    x[0]=x0
    x_minus = np.mat(np.zeros(shape=(n_time, n_state)))
    x_minus[0] = x0
    
    # factor errors
    P = [0 for i in range(n_time)]
    P[0] = P0
    P_minus = [0 for i in range(n_time)]
    P_minus[0] = P0
    
    # Kalman gains
    K = [0 for i in range(n_time)]
    
    "Kalman Filter"
    for i in range(1,n_time):
        # check if z is available
        ix = np.where(1-np.isnan(z[i]))[1]
        z_t = z[i,ix]
        H_t = H[ix]
        R_t = R[ix][:,ix]

        "prediction step"
        #x_k = F*x_{k-1} + B*u_k
        x_minus[i] = A.dot(x[i-1].T).T + B.dot(u[i].T).T
        P_minus[i] = A.dot(P[i-1]).dot(A.T) + Q
        
        "update step"
        temp = H_t.dot(P_minus[i]).dot(H_t.T) + R_t
        K[i] = P_minus[i].dot(H_t.T).dot(temp.I)
        P[i] = P_minus[i] - K[i].dot(H_t).dot(P_minus[i])
        x[i] = (x_minus[i].T + K[i].dot(z_t.T-H_t.dot(x_minus[i].T))).T
    
    x = pd.DataFrame(data=x, index=Z.index, columns=state_names)
    x_minus = pd.DataFrame(data=x_minus, index=Z.index, columns=state_names)

    
    return KalmanFilterResultsWrapper(x_minus=x_minus, x=x, z=Z, Kalman_gain=K, P=P, P_minus=P_minus, state_names = state_names, A=A)

def FIS(res_KF):
    N = len(res_KF.x.index)
    n_state = len(res_KF.x.columns)
    x = np.mat(res_KF.x)
    x_minus = np.mat(res_KF.x_minus)
    P = res_KF.P
    P_minus = res_KF.P_minus
    
    
    x_sm = np.mat(np.zeros(shape=(N, n_state)))
    x_sm[N-1] = x[N-1]
    
    P_sm = [0 for i in range(N)]
    P_sm[N-1] = P[N-1]
    
    J = [0 for i in range(N)]
    
    for i in reversed(range(N-1)):
        J[i] = P[i].dot(res_KF.A).dot(P_minus[i+1].I)
        P_sm[i] = P[i] - J[i].dot(P_minus[i+1]-P_sm[i+1]).dot(J[i].T)
        x_sm[i] = (x[i].T + J[i].dot(x_sm[i+1].T-x_minus[i+1].T)).T
    
    x_sm = pd.DataFrame(data=x_sm, index=res_KF.x.index, columns=res_KF.x.columns)
    
    return SKFResultsWrapper(x_sm=x_sm, P_sm=P_sm,z=res_KF.z)
        
class SKFResultsWrapper():
    def __init__(self, x_sm, P_sm, z):
        self.x_sm = x_sm
        self.P_sm = P_sm
        self.z = z
    

    
def EMstep(res_SKF, n_shocks):
    f = res_SKF.x_sm
    y = res_SKF.z
    
    Lambda = calculate_factor_loadings(y, f)
    A = calculate_prediction_matrix(f)
    B, Q = calculate_shock_matrix(f, A, n_shocks) 
    
    resid = (np.mat(y).T - Lambda.dot(np.mat(f).T)).T
    temp = [resid[:,i] for i in range(len(y.columns))]
    R = np.diag(np.diag(np.cov(resid.T)))
    
    return EMstepResultsWrapper(Lambda=Lambda, A=A, B=B, Q=Q, R=R, x_sm=f, z=y)
    
class EMstepResultsWrapper():
    def __init__(self, Lambda, A, B, Q, R, x_sm, z):
        self.Lambda = Lambda
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x_sm = x_sm
        self.z = z

    