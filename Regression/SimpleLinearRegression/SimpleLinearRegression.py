#!/usr/bin/env python
# coding: utf-8

# # Import the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


#  # Linear Regression 
#  ## y = b0 + b1X ( where b0 is the intercept and b1 is the slope, X is independent variable)

# In[5]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[8]:


y_pred = regressor.predict(X_test)


# ## Visualize the training set

# In[15]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# ## Visualize the test set

# In[16]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:




