#!/usr/bin/env python
# coding: utf-8

# In[50]:


get_ipython().system('pip install apyori')


# ## Import the Libraries
# 

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[52]:


dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)


# In[53]:


transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])


# ## Train the Apriori Model

# In[54]:


from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)


# ## Visualize the Results 

# ## Displaying the results coming from the Apriori Function

# In[55]:


results = list(rules)
results


# ## Organise the Data properly

# In[56]:


def inspect(results):
    lhs        = [tuple(result[2][0][0])[0] for result in results]
    rhs        = [tuple(result[2][0][1])[0] for result in results]
    supports   = [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lift       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidence, lift))
    
results_in_dataframe = pd.DataFrame(inspect(results), columns = ['LHS', 'RHS', 'SUPPORTS', 'CONFIDENCE', 'LIFT'])
print(results_in_dataframe)


# ## Sort the Results

# In[57]:


results_in_dataframe.nlargest(n = 10, columns = 'LIFT')


# In[ ]:




