#!/usr/bin/env python
# coding: utf-8

# ## Include Necessary Libraries

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[28]:


dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values


# ## Determine the optimal number of clusters

# In[29]:


## In KMeans we used Elbow method, here we will Hierarchical clustering method, which uses Dendrograms to determine the number of clusters
import scipy.cluster.hierarchy as sch
dendrogram  = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eculidean Distance')
plt.show()


# ## Make the Model

# In[30]:


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5, linkage = 'ward', metric = 'euclidean')
y_hc = cluster.fit_predict(X)
print(y_hc)


# ## Visualize the Model

# In[31]:


plt.scatter(X[y_hc == 0 , 0], X[y_hc == 0 , 1], s = 10, color = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1 , 0], X[y_hc == 1 , 1], s = 10, color = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2 , 0], X[y_hc == 2 , 1], s = 10, color = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3 , 0], X[y_hc == 3 , 1], s = 10, color = 'yellow', label = 'Cluster 4')
plt.scatter(X[y_hc == 4 , 0], X[y_hc == 4 , 1], s = 10, color = 'cyan', label = 'Cluster 5')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Cluster of Customers')
plt.legend()


# In[ ]:




