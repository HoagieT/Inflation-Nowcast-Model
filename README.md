# Inflation Nowcast
A dynamic factor model that forecasts inflation, i.e. CPI, PPI, in China

This code implements the nowcasting framework described in "[Macroeconomic Nowcasting and Forecasting with Big Data](https://www.newyorkfed.org/research/staff_reports/sr830.html)" by Brandyn Bok, Daniele Caratelli, Domenico Giannone, Argia M. Sbordone, and Andrea Tambalotti, *Staff Reports 830*, Federal Reserve Bank of New York (prepared for Volume 10 of the *Annual Review of Economics*).

**Note:** The vintage example files (Vintage_CPI.py, Vintage_PPI.py) require installation of Wind Database and Wind Python API. Users may use their own data source as well.

## Download instructions

Download the code as a ZIP file by clicking the green 'Clone or download' button and selecting 'Download ZIP'.


## File and folder description

* `DiscreteKalmanFilter.py` : Discrete Kalman Filter algorithm which is applied to estimate parameters in Dynamic Factor Model as well as dealing with unbalanced dataset
* `DynamicFactorModel.py` : Dynamic Factor Model module, which is the main body of nowcasting algorithm
* `Functions.py` : Miscellaneous functions
* `Vintage_PPI.py` : Vintage generator for PPI related data, Wind API required
* `Nowcast_PPI.py` : To nowcast PPI, Vintage_PPI.py must be run first
