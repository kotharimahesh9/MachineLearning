#!/usr/bin/env python
# coding: utf-8

# In[37]:


# IMPORT THE LIBRARIES


# In[38]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[39]:


# IMPORT THE DATASET


# In[40]:


dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(Y)


# # Handle Missing Values

# In[41]:


from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)


# # One Hot Encoding

# In[42]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# In[43]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)


# # Test Train Split

# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[45]:


print(X_train)


# In[46]:


print(X_test)


# In[47]:


print(y_train)


# In[48]:


print(y_test)


# # Feature Scaling

# In[49]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3: ] = sc.transform(X_test[:, 3:])


# In[50]:


print(X_train)


# In[51]:


print(X_test)


# In[ ]:




