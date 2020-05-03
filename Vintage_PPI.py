# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:04:01 2020

@author: 029046
"""
from Functions import *
from WindPy import w
w.start(waitTime=60) 
w.isconnected()

dates = pd.date_range(start=date(2014,11,9), end=date(2020,5,10), freq='W')
start_date='2014-11-9'
extract_date='2020-5-10'

"Mothly data"
temp=w.edb("M0049160,M0017134", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times,columns=['PPI_MoM','PMI_RawMaterialPrice'])
data_monthly = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_monthly.index)):
    sub = data_raw[(data_raw.index>=data_monthly.index[i-1].date())&(data_raw.index<data_monthly.index[i].date())].mean()
    data_monthly.iloc[i-1,]=sub.values

"Energy"
temp=w.edb("S5103941,S5103943,S5914475,S5914476,S5125926,S5103933,S5103935,S5914486,S5914482,S5914481", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Ethanol97','Ethanol93','LNG','LPG','Diesel','Gasoline97','Gasoline93','Coke','BlendedCoal','AnthraciteCoal'])
data_energy = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_energy.index)):
    sub = data_raw[(data_raw.index>data_energy.index[i-1].date())&(data_raw.index<=data_energy.index[i].date())].mean()
    data_energy.iloc[i,]=sub.values
    
"Chemicals"
temp=w.edb("S5914465,S5914466,S5914467,S5914468,S5914469,S5914470,S5914471,S5914472,S5914499,S5470301", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Sulfate','LiquidSoda','Methanol','Benzene','Styrene','LLDPE','Phenylpropene','PVC','Urea','PTA'])
data_chemical = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_chemical.index)):
    sub = data_raw[(data_raw.index>data_chemical.index[i-1].date())&(data_raw.index<=data_chemical.index[i].date())].mean()
    data_chemical.iloc[i,]=sub.values
    
"Metals"
temp=w.edb("S0105511,S0105512,S0105513,S0105514,S0105516,S0105517", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Cu','Al','Pb','Zn','Sn','Ni'])
data_metal = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_metal.index)):
    sub = data_raw[(data_raw.index>data_metal.index[i-1].date())&(data_raw.index<=data_metal.index[i].date())].mean()
    data_metal.iloc[i,]=sub.values

"Steel Product"
temp=w.edb("S5914455,S5914456,S5914457,S5914458,S5914459,S5914460", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Rebar','WireRod','Plate','HotRolledSheet','SeamlessSteelPipe','AngleSteel'])
data_steel = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_steel.index)):
    sub = data_raw[(data_raw.index>data_steel.index[i-1].date())&(data_raw.index<=data_steel.index[i].date())].mean()
    data_steel.iloc[i,]=sub.values
"""    
temp=w.edb("S0179664,S5711154,S5711155,S5711153,S0179661", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Rebar','I-Steel','RoundSteel','EqualAngleSteel','HotRolledPlate'])
data_steel = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_steel.index)):
    sub = data_raw[(data_raw.index>data_steel.index[i-1].date())&(data_raw.index<=data_steel.index[i].date())].mean()
    data_steel.iloc[i,]=sub.values
"""
"Food Product"
temp=w.edb("S5914491,S5914492,S5914493,S5914494,S5914495,S5914496,S5914498", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Rice','Wheat','Corn','Cotton','Pig','Soybean','Peanut'])
data_food = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_food.index)):
    sub = data_raw[(data_raw.index>data_food.index[i-1].date())&(data_raw.index<=data_food.index[i].date())].mean()
    data_food.iloc[i,]=sub.values

"Construction & Furniture"
temp=w.edb("S5914515,S5919833,S5907373", start_date, extract_date)
data_raw = pd.DataFrame(data=np.array(temp.Data).T, index=temp.Times, columns=['Cement','Concrete','Glass'])
data_construction = pd.DataFrame(data=np.nan,index=dates,columns=data_raw.columns)
for i in range(1,len(data_construction.index)):
    sub = data_raw[(data_raw.index>data_construction.index[i-1].date())&(data_raw.index<=data_construction.index[i].date())].mean()
    data_construction.iloc[i,]=sub.values

"Vintage"
data_weekly=data_energy.join(data_chemical).join(data_metal).join(data_steel).join(data_food).join(data_construction)

temp = transform_data(data=data_weekly, method='MoM')
vintage_PPI=data_monthly.join(temp)[:-1]

