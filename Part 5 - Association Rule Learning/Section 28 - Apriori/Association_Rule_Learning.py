# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:57:22 2019

@author: laksh
"""

# Apriori Algorithm

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:33:08 2019

@author: laksh
"""
#Data Preprocessing Template

# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support =0.003, min_confidence =0.2, min_lift =3, min_length = 2)

# Visualising the results
results = list(rules)

for i in results:
    print(i)
    print('**********')

















