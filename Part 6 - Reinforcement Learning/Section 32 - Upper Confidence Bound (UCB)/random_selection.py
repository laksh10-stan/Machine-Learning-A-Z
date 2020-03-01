# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 03:35:38 2019

@author: laksh
"""

# Random Select Strategy
# Click through rate --> CTR
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for i in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[i, ad]
    total_reward += reward

# Visualising the results --Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()    
    
