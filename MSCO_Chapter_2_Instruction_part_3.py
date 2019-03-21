import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# regression models before excel
df = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.4_table_2.9', header=0, index_col=None)
# print(df)

import statsmodels.api as sm
X = sm.add_constant(df['t'])

results = sm.OLS(df['Dt'], X).fit()
print("\n StatsModels OLS summary: \n", results.summary(), "\n StatsModels OLS parameters: \n", results.params)

from sklearn.linear_model import LinearRegression
linmodel = LinearRegression().fit(X,df['Dt'])
print("\n Scikit-learn intercept and coefficient: \n",linmodel.intercept_, linmodel.coef_[1])

from numpy import polyfit
polmodel = polyfit(df['t'],df['Dt'],1)
print("\n NumPy 1st-degree polyfit \n", polmodel)

from scipy.stats import linregress
scimodel = linregress(df['t'],df['Dt'])
print("\n SciPy linear regression: \n", scimodel.slope, scimodel.intercept)

from scipy.optimize import curve_fit
def func(m,x,c):
    y = m*x+c
    return y
popt, pcov = curve_fit(func,df['t'],df['Dt'])
print("\n SciPy curve_fit: \n", popt)

from numpy.linalg import lstsq
lstsqmod = lstsq(X,df['Dt'], rcond=None)
print("\n NumPy least-squares: \n", lstsqmod[0])

y = np.array([df['Dt']]).T
beta_hat = np.dot(np.linalg.inv(np.dot(X.T,X)),(np.dot(X.T,y)))
print("\n Beta-hat intercept: \n", beta_hat[0],"\n Beta-hat slope: \n", beta_hat[1])

pseudo_beta_hat = np.dot(np.linalg.pinv(np.dot(X.T,X)),(np.dot(X.T,y)))
print("\n Pseudo-beta-hat intercept: \n", pseudo_beta_hat[0],"\n Pseudo-beta-hat slope: \n", pseudo_beta_hat[1])

intercept = beta_hat[0][0]
slope = beta_hat[1][0]

def linRegPred(period):
    y = intercept+slope*period
    return y

new_pred = []
for i in range (1, len(df['t'])+11):
    pred = np.round(linRegPred(i),2)
    new_pred.append([i,pred])
new_pred = pd.DataFrame(new_pred, columns=['t','Ft'])

df2 = pd.merge(df,new_pred,how='outer',on=['t'])
print(df2)

# This plots the demand and the linear regression algorithm's predicted demand
plt.ylabel('Widget Demand (000s)');
plt.ylim(0,75);
plt.axes().yaxis.grid(linestyle=':');
plt.xlabel('Time Period');
plt.xlim(0,18);
plt.title('Widget Demand Data');
plt.plot(df2['t'], df2['Ft'], marker='D');
plt.plot(df2['t'], df2['Dt'], marker='o');
plt.legend();
plt.show();

# # reg model from fig 2.13
df = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.4_figure_2.13', header=2, index_col=None, nrows=10, usecols=[0,1])

# This calculates the function intercepts
import statsmodels.api as sm
X = sm.add_constant(df['Month'])
y = np.array([df['Demand']]).T
beta_hat = np.dot(np.linalg.inv(np.dot(X.T,X)),(np.dot(X.T,y)))
intercept = beta_hat[0][0]
slope = beta_hat[1][0]

# This defines the function
def linRegPred(period):
    y = intercept+slope*period
    return y

# This calls the function to create forecasted values
new_pred = []
for i in range (1, len(df['Month'])+11):
    pred = np.round(linRegPred(i),2)
    new_pred.append([i,pred])
new_pred = pd.DataFrame(new_pred, columns=['Month','Forecast'])

# This merges the new dataframe (the one with forecasts) with the old
df2 = pd.merge(df,new_pred,how='outer',on=['Month'])
print(df2)

# This plots the demand and the linear regression algorithm's predicted demand
plt.ylabel('Widget Demand (000s)');
plt.ylim(0,120);
plt.axes().yaxis.grid(linestyle=':');
plt.xlabel('Month');
plt.xlim(0,18);
plt.title('Widget Demand Data');
plt.plot(df2['Month'], df2['Forecast'], marker='D');
plt.plot(df2['Month'], df2['Demand'], marker='o');
plt.legend();
plt.show();
