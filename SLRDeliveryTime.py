import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

delivery1=pd.read_csv("C:/My Files/Excelr/03 - Simple Linear Regression/Assignment/delivery_time.csv")
delivery1.corr()

import seaborn as sn
sn.pairplot(delivery1)
delivery1.columns
delivery = pd.DataFrame(delivery1)
delivery.rename(columns = {'Delivery Time':'DeliveryTime'}, inplace = True)
delivery.rename(columns = {'Sorting Time':'SortingTime'}, inplace = True)

import statsmodels.formula.api as sfa
model=sfa.ols('DeliveryTime~SortingTime',data=delivery).fit()
sm.graphics.plot_partregress_grid(model)
model.summary()
print(model.conf_int(0.05))

dtime_predicted = model.predict(delivery[['SortingTime']])
dtime_predicted
rsq = sfa.ols('DeliveryTime~SortingTime',data=delivery).fit().rsquared  
rsq

plt.scatter(delivery.DeliveryTime,dtime_predicted,c="r")
plt.xlabel("observed_values");plt.ylabel("fitted_values")

plt.hist(model.resid_pearson)

import pylab          
import scipy.stats as st
st.probplot(model.resid_pearson,dist="norm",plot=pylab)