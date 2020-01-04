import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

carsData = pd.read_csv("C:/My Files/Excelr/04 - Multiple Linear Regression/Assignment/cars.csv")
carsData.head()
carsData.tail()
carsData.describe()
carsData.corr()
import seaborn as sn
sn.pairplot(carsData)
carsData.columns

import statsmodels.formula.api as sfa
model1 = sfa.ols('MPG~WT+VOL+SP+HP',data=carsData).fit()
model1.params
model1.summary()

#Check for co linearity between wt and vol
subm1 = sfa.ols('MPG~WT',data=carsData).fit()
subm1.summary()
subm2 = sfa.ols('MPG~VOL',data=carsData).fit()
subm2.summary()
subm3 = sfa.ols('WT~VOL',data=carsData).fit()
subm3.summary()
subm4 = sfa.ols('MPG~VOL+WT',data=carsData).fit()
subm4.summary()

#check for influence plot
fig = sm.graphics.influence_plot(model1);
cars_new_data = carsData.drop(carsData.index[[76,70]],axis=0)
model2 = sfa.ols('MPG~WT+VOL+SP+HP',data=cars_new_data).fit()
model2.summary()
print(model2.conf_int(0.01))

# Predicted values of MPG 
mpg_predicted = model2.predict(cars_new_data[['WT','VOL','HP','SP']])
mpg_predicted

# calculating VIF's values of independent variables
rsq_hp = sfa.ols('HP~WT+VOL+SP',data=cars_new_data).fit().rsquared  
vif_hp = 1/(1-rsq_hp)

rsq_wt = sfa.ols('WT~HP+VOL+SP',data=cars_new_data).fit().rsquared  
vif_wt = 1/(1-rsq_wt)

rsq_vol = sfa.ols('VOL~WT+SP+HP',data=cars_new_data).fit().rsquared  
vif_vol = 1/(1-rsq_vol)

rsq_sp = sfa.ols('SP~WT+VOL+HP',data=cars_new_data).fit().rsquared  
vif_sp = 1/(1-rsq_sp)

d1= {'variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
vif_dframe = pd.DataFrame(d1)
vif_dframe

# Added varible plot
sm.graphics.plot_partregress_grid(model2)

#final model
finalModel = sfa.ols('MPG~VOL+SP+HP',data=cars_new_data).fit()
finalModel.summary()


mpg_pred = finalModel.predict(cars_new_data)
mpg_pred
finalModel.resid_pearson

# Observed values vs Fitted values
plt.scatter(cars_new_data.MPG,mpg_pred,c="r");
plt.xlabel("observed_values");plt.ylabel("fitted_values")

plt.hist(finalModel.resid_pearson)

import pylab          
import scipy.stats as st
st.probplot(finalModel.resid_pearson, dist="norm", plot=pylab)
