# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:17:37 2020

@author: 029046
"""

from WindPy import w
from DiscreteKalmanFilter import *
from Functions import *
from DynamicFactorModel import *

w.start(waitTime=60)
w.isconnected()


plt.rcParams['figure.figsize'] = (9, 6)
data = vintage_PPI
data_sm = DataInterpolation(data, 0, len(data.index), 'linear').dropna(axis=0,how='any')
data_dev = data_sm-data_sm.mean()


dfm_em = DFM_EMalgo(data_sm, 10, 2, 3)

#error = pd.DataFrame(data=rand_Matrix(len(data.index), 2),columns=['shock1','shock2'],index=data.index)
error = pd.DataFrame(data=np.zeros(shape=(len(data.index), 2)),columns=['shock1','shock2'],index=data.index)
kf = KalmanFilter(Z=data.iloc[6:]-data_sm.mean(), U=error, A=dfm_em.A, B=dfm_em.B, H=dfm_em.Lambda, state_names=dfm_em.x.columns, x0=dfm_em.x.iloc[0], P0=calculate_covariance(dfm_em.x), Q=dfm_em.Q, R=dfm_em.R)

predict=RevserseTranslate(kf.x, data_sm.mean(), dfm_em.Lambda, data_sm.columns)

plot_compare(data_sm.iloc[:,0], predict.iloc[:,0], 'PPI MoM',fig_size=[12,6], line_width=2.0,font_size='xx-large')
