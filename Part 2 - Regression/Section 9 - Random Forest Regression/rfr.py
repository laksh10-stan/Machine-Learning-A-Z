# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:18:01 2019

@author: laksh
"""

#Random forest Regression
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 00:11:21 2019

@author: laksh
"""

#Regression Template

#Data Preprocessing Template

#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

'''
#Splitting dataset into training set and Test set
#Doesn't make much sense because data is insufficient

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''
'''
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

#Fitting Random forest Regression Model to the dataset
#Create your Random forest Regression here
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)
#Predicting a new result Random forest Regression
y_pred = regressor.predict([[6.5]])

'''
#Visualising the Random forest Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Random forest Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
'''

#Visualising the Random forest Regression Results (for higher resolution and smooth curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random forest Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
