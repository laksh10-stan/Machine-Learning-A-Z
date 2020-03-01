# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:59:28 2019

@author: laksh
"""
#Artificial Neural Network (PART 1)

#Data Preprocessing Template

#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

#Encoding categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Dummy Encoding
 
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Removing Dummy Variable Trap
X = X[:, 1:]

#Splitting dataset into training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Part 2 Building ANN

# Import the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the first layer and the first Hidden Layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))  # Outdated (Warning but no error)
#classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim = 11)) # Updated

# Second Hidden Layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the Output Layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 Making the predictions and Evaluations


#Predicting Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)















