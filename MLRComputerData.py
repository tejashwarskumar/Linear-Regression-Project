import pandas as pd
computer = pd.read_csv("C:/My Files/Excelr/04 - Multiple Linear Regression/Assignment/Computer_Data.csv")
computer.columns
computer_dummies = pd.get_dummies(computer[["cd","multi","premium"]])
computer.drop(["cd","multi","premium"],inplace=True,axis = 1)
computer.drop(["Unnamed: 0"],inplace=True,axis=1)
computer = pd.concat([computer,computer_dummies],axis=1)
computer.describe()

import seaborn as sb
sb.pairplot(computer)
computer.corr()
import statsmodels.formula.api as sfa;

model1 = sfa.ols("price ~ speed+hd+ram+screen+ads+trend+cd_no+cd_yes+multi_no+multi_yes+premium_no+premium_yes",data=computer).fit()
model1.summary()
predict_com = model1.predict(computer)
model1.resid_pearson
import matplotlib.pyplot as plt
plt.scatter(computer.price,predict_com,c="r");
plt.xlabel("observed_values");plt.ylabel("fitted_values")

plt.hist(model1.resid_pearson)

import pylab          
import scipy.stats as st
st.probplot(model1.resid_pearson,dist="norm",plot=pylab)
