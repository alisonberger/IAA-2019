#!/usr/bin/env python
# coding: utf-8

# In[37]:


#Data Mining Homework #1 - Alison Berger

#import needed libraries
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml

#read in the data
df=pd.read_csv("C:/Users/alb27/Documents/Data Mining/restaurantData.csv")
df.head(8)


# In[33]:


#visualize the data by frequencies 
sns.countplot(x = 'order', data = df, order = df['order'].value_counts().iloc[:10].index)
plt.xticks(rotation=90)

#most popular main dish:pork-tendorloin
#most popular side dish:roasted root veg
#most popular wine: cantina pinot bianco


# In[42]:


df = df.groupby(['order','orderNumber']).size().reset_index(name='count')
basket = (df.groupby(['orderNumber', 'order'])['count']
          .sum().unstack().reset_index().fillna(0)
          .set_index('orderNumber'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
basket_sets = basket.applymap(encode_units)

frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift")
rules.sort_values('lift', ascending = False, inplace = True)
rules.head(10)

#sorted by lift 
#top item - if a customer orders raost chicken, they are 2.75 times more likely to also purchase duckhorn chardonnay


# In[36]:





# In[ ]:




