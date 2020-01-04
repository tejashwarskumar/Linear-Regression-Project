import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

calories1=pd.read_csv("C:/My Files/Excelr/03 - Simple Linear Regression/Assignment/calories_consumed.csv")
calories1.corr()

import seaborn as sn
sn.pairplot(calories1)
calories1.columns
calories = pd.DataFrame(calories1)
calories.rename(columns = {'Weight gained (grams)':'Weightgainedgrams'}, inplace = True)
calories.rename(columns = {'Calories Consumed':'CaloriesConsumed'}, inplace = True)

import statsmodels.formula.api as sfa
model=sfa.ols('Weightgainedgrams~CaloriesConsumed',data=calories).fit()
sm.graphics.plot_partregress_grid(model)
model.summary()
print(model.conf_int(0.05))

weight_predicted = model.predict(calories[['CaloriesConsumed']])
weight_predicted
rsq = sfa.ols('Weightgainedgrams~CaloriesConsumed',data=calories).fit().rsquared  
rsq

plt.scatter(calories.Weightgainedgrams,weight_predicted,c="r")
plt.xlabel("observed_values");plt.ylabel("fitted_values")

plt.hist(model.resid_pearson)

import pylab          
import scipy.stats as st
st.probplot(model.resid_pearson,dist="norm",plot=pylab)