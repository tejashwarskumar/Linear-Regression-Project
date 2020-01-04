import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

salary1=pd.read_csv("C:/My Files/Excelr/03 - Simple Linear Regression/Assignment/Salary_Data.csv")
salary1.corr()

import seaborn as sn
sn.pairplot(salary1)
salary1.columns
salary = pd.DataFrame(salary1)

import statsmodels.formula.api as sfa
model=sfa.ols('Salary~YearsExperience',data=salary).fit()
sm.graphics.plot_partregress_grid(model)
model.summary()
print(model.conf_int(0.05))

salary_predicted = model.predict(salary[['YearsExperience']])
salary_predicted
rsq = sfa.ols('Salary~YearsExperience',data=salary).fit().rsquared  
rsq

plt.scatter(salary.Salary,salary_predicted,c="r")
plt.xlabel("observed_values");plt.ylabel("fitted_values")

plt.hist(model.resid_pearson)

import pylab          
import scipy.stats as st
st.probplot(model.resid_pearson,dist="norm",plot=pylab)