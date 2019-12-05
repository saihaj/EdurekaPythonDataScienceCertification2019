#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os 
os.chdir('D:\\Trainings\\R and Python Classes\\Machine Learning A-Z\\Part 5 - Association Rule Learning\\Section 28 - Apriori\\Apriori-Python\\Apriori_Python\\')


# In[5]:


dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)


# In[6]:


dataset


# In[7]:


transactions = []


# In[9]:


for i in  range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])


# In[11]:


from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift = 3, min_length=2)


# In[12]:


results = list(rules)


# In[14]:


pd.DataFrame(results)

