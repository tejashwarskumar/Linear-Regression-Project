import pandas as pd
toyota = pd.read_csv("C:/My Files/Excelr/04 - Multiple Linear Regression/Assignment/ToyotaCorolla.csv")
toyota.describe()
toyota.columns

import seaborn as sb
sb.pairplot(toyota)
cor = toyota.corr()

import statsmodels.formula.api as sfa
model1 = sfa.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit()
model1.summary()

submodel1 = sfa.ols('Price~Doors',data=toyota).fit()
submodel1.summary()
submodel2 = sfa.ols('Price~cc',data=toyota).fit()
submodel2.summary()

submodel3 = sfa.ols('Doors~cc',data=toyota).fit()
submodel3.summary()
 
import statsmodels.api as st;
fig = st.graphics.influence_plot(model1)

new_toyotaData = toyota.drop(toyota.index[[80]],axis=0)

model2 = sfa.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=new_toyotaData).fit()
model2.summary()

#VIF for all variables
rsq_age = sfa.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit().rsquared  
vif_age = 1/(1-rsq_age)

rsq_km = sfa.ols('KM ~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit().rsquared  
vif_km = 1/(1-rsq_km)

rsq_hp = sfa.ols('HP ~KM+Age_08_04+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit().rsquared  
vif_hp = 1/(1-rsq_hp)

rsq_cc = sfa.ols('cc~KM+HP+Age_08_04+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit().rsquared  
vif_cc = 1/(1-rsq_cc)

rsq_doors = sfa.ols('Doors ~KM+HP+cc+Age_08_04+Gears+Quarterly_Tax+Weight',data=toyota).fit().rsquared  
vif_doors = 1/(1-rsq_doors)

rsq_gears = sfa.ols('Gears ~KM+HP+cc+Age_08_04+Doors+Quarterly_Tax+Weight',data=toyota).fit().rsquared  
vif_gears = 1/(1-rsq_gears)

rsq_qt = sfa.ols('Quarterly_Tax  ~KM+HP+cc+Age_08_04+Doors+Gears+Weight',data=toyota).fit().rsquared  
vif_qt = 1/(1-rsq_qt)

rsq_wt = sfa.ols('Weight ~KM+HP+cc+Age_08_04+Doors+Gears+Quarterly_Tax',data=toyota).fit().rsquared  
vif_wt = 1/(1-rsq_wt)

st.graphics.plot_partregress_grid(model2)

finalModel = sfa.ols('Price~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=new_toyotaData).fit()
finalModel.summary()

predictData = finalModel.predict(new_toyotaData)
predictData
finalModel.resid_pearson

import matplotlib.pyplot as plt;
plt.scatter(new_toyotaData.Price,predictData,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

plt.hist(finalModel.resid_pearson)

import pylab
import scipy.stats as st
st.probplot(finalModel.resid_pearson,dist="norm", plot=pylab)
