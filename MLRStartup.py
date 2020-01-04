import pandas as pd

startup = pd.read_csv("C:/My Files/Excelr/04 - Multiple Linear Regression/Assignment/50_Startups.csv")
startup.describe()
startup.columns = ['RD_Spend', 'Administration', 'Marketing_Spend','State','Profit']

startup_dummies = pd.get_dummies(startup[['State']])
startup.drop(['State'],inplace=True,axis = 1)
startup = pd.concat([startup,startup_dummies],axis=1)
startup.columns = ['RD_Spend', 'Administration', 'Marketing_Spend', 'Profit','State_California', 'State_Florida', 'State_New_York']
import seaborn as sb
sb.pairplot(startup)
startup.corr()

import statsmodels.formula.api as sfm;

model1 = sfm.ols("Profit ~ RD_Spend+Administration+Marketing_Spend+State_California+State_Florida+State_New_York",data=startup).fit()
model1.summary()

subm1 = sfm.ols("Profit ~ Administration",data=startup).fit();
subm1.summary() 
subm2 = sfm.ols("Profit ~ Marketing_Spend",data=startup).fit();
subm2.summary()
subm3 = sfm.ols("Profit ~ Administration+Marketing_Spend",data=startup).fit();
subm3.summary()

#check for influence plot
import statsmodels.api as sm
fig = sm.graphics.influence_plot(model1);

startup_new_data = startup.drop(startup.index[[48,49]],axis=0)
model_influence = sfm.ols("Profit ~ RD_Spend+Administration+Marketing_Spend+State_California+State_Florida+State_New_York",data=startup_new_data).fit()
model_influence.summary()
 
# calculating VIF's values of independent variables
rsq_rd = sfm.ols('RD_Spend~Administration+Marketing_Spend',data=startup_new_data).fit().rsquared  
vif_rd = 1/(1-rsq_rd)

rsq_ad = sfm.ols('Administration~RD_Spend+Marketing_Spend',data=startup_new_data).fit().rsquared  
vif_ad = 1/(1-rsq_ad)

rsq_ms = sfm.ols('Marketing_Spend~RD_Spend+Administration',data=startup_new_data).fit().rsquared  
vif_ms = 1/(1-rsq_ms)

pd.set_option('display.expand_frame_repr', False)

#check for av plot
sm.graphics.plot_partregress_grid(model_influence)

final_model = sfm.ols('Profit~RD_Spend+Marketing_Spend+State_California+State_Florida+State_New_York',data=startup_new_data).fit()
final_model.summary()

profit_pred = final_model.predict(startup_new_data)
profit_pred
final_model.resid_pearson

import matplotlib.pyplot as plt
plt.scatter(startup_new_data.Profit,profit_pred,c="r");
plt.xlabel("observed_values");plt.ylabel("fitted_values")

plt.hist(final_model.resid_pearson)

import pylab          
import scipy.stats as st
st.probplot(final_model.resid_pearson, dist="norm", plot=pylab)
