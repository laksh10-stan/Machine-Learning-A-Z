# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:04:16 2019

@author: laksh
"""
#Data Preprocessing Template
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# Dummy Encoding 
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable trap
X = X[:, 1:]

#Splitting dataset into training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

'''
#~~~~~NOT WORKING ~~~~#
#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int) , values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#regressor_OLS = sm.ols(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
'''
#Building the optimal model using Backward Elimination
import statsmodels.regression.linear_model as lm
X = np.append(arr = np.ones((50, 1)).astype(int) , values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

'''
#Building the optimal model using Backward Elimination
import statsmodels.api as smf
X3 = np.append(arr = np.ones((50, 1), dtype=np.int) , values = X, axis = 1)
X_opt3 = X3[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS3 = smf.OLS(y, X_opt3).fit()
regressor_OLS3.summary()

'''











































