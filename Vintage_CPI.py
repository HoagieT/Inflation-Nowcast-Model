# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:53:12 2020

@author: 029046
"""

from Functions import *
from WindPy import w
w.start(waitTime=60)
w.isconnected()

dates = pd.date_range(start=date(2015,12,6), end=date(2020,5,8), freq='W')
start_date='2015-12-6'
extract_date='2020-5-8'

"Mothly data"
temp=w.edb("M0000705,M0000706", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times,columns=['CPI_MoM','CPI_Food_MoM'])
data_monthly = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_monthly.index)):
    sub = data_raw[(data_raw.index>=data_monthly.index[i-1].date())&(data_raw.index<data_monthly.index[i].date())].mean()
    data_monthly.iloc[i-1,]=sub.values
    
"Food"
temp=w.edb("S5065106,S5065107,S5065108,S5065109,S5065110,S5065111,S5065112,S5065114,S5065115,S5065116,S5010204", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Pork','Beef','Lamb','Egg','Chicken','Vegetable','Fruit','Carp','SilverCarp','Cutlass','Salt'])
data_food = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_food.index)):
    sub = data_raw[(data_raw.index>data_food.index[i-1].date())&(data_raw.index<=data_food.index[i].date())].mean()
    data_food.iloc[i,]=sub.values

"Transportation and Communication"
temp=w.edb("S5103935,S5103933,S6500615,S6006774,S6006789", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Gasoline93','Gasoline97','Computer','CommunicationWire','TransportationSafety'])
data_transcomm = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_transcomm.index)):
    sub = data_raw[(data_raw.index>data_transcomm.index[i-1].date())&(data_raw.index<=data_transcomm.index[i].date())].mean()
    data_transcomm.iloc[i,]=sub.values

"Medical and Healthcare"
temp=w.edb("S0106873,S0106875,S0106877,S0106883,S6203722,S6263057,S6273295", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['VitaminA','VitaminE','VitaminD','FolicAcid','Lysine','Methionine','Poison'])
data_medical = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_medical.index)):
    sub = data_raw[(data_raw.index>data_medical.index[i-1].date())&(data_raw.index<=data_medical.index[i].date())].mean()
    data_medical.iloc[i,]=sub.values

"Tobacco and Alcohol"
temp=w.edb("S5201107,S5201111,S5201091,S5201095,S5201083", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Budweiser','Harbin','Zhangyu','Changcheng','Wuliangye'])
data_alcohol = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_alcohol.index)):
    sub = data_raw[(data_raw.index>data_alcohol.index[i-1].date())&(data_raw.index<=data_alcohol.index[i].date())].mean()
    data_alcohol.iloc[i,]=sub.values

"Cultural Products"
temp=w.edb("S0161486,S0161478,S0161474", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['OfficeSupplies','Paper','Writing'])
data_culture = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_culture.index)):
    sub = data_raw[(data_raw.index>data_culture.index[i-1].date())&(data_raw.index<=data_culture.index[i].date())].mean()
    data_culture.iloc[i,]=sub.values

"Cloth"
temp=w.edb("S0049600,S0049606,S0049610,S0049624,S0049629", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Material','Fabric','Garment ','HomeTextile','Accessories'])
data_cloth = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_cloth.index)):
    sub = data_raw[(data_raw.index>data_cloth.index[i-1].date())&(data_raw.index<=data_cloth.index[i].date())].mean()
    data_cloth.iloc[i,]=sub.values



"Vintage"
data_weekly=data_food.join(data_transcomm).join(data_medical).join(data_alcohol).join(data_culture).join(data_cloth)

temp = transform_data(data=data_weekly, method='MoM')
vintage_CPI=data_monthly.join(temp)[6:-1]
