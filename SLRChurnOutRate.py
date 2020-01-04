import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

churn1=pd.read_csv("C:/My Files/Excelr/03 - Simple Linear Regression/Assignment/emp_data.csv")
churn1.corr()

import seaborn as sn
sn.pairplot(churn1)
churn1.columns
churn = pd.DataFrame(churn1)

import statsmodels.formula.api as sfa
model=sfa.ols('Churn_out_rate~Salary_hike',data=churn).fit()
sm.graphics.plot_partregress_grid(model)
model.summary()
print(model.conf_int(0.05))

churn_predicted = model.predict(churn[['Salary_hike']])
churn_predicted
rsq = sfa.ols('Churn_out_rate~Salary_hike',data=churn).fit().rsquared  
rsq

plt.scatter(churn.Churn_out_rate,churn_predicted,c="r")
plt.xlabel("observed_values");plt.ylabel("fitted_values")

plt.hist(model.resid_pearson)

import pylab          
import scipy.stats as st
st.probplot(model.resid_pearson,dist="norm",plot=pylab)