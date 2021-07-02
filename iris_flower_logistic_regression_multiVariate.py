#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("iris.csv")
df.isnull()


# In[3]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[['sepal_length','sepal_width','petal_length','petal_width']],df.species,train_size=0.7)


# In[4]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
X_test


# In[5]:


model.fit(X_train,y_train)


# In[6]:


model.score(X_test,y_test)


# In[7]:


model.predict(X_test)


# In[8]:


model.coef_


# In[9]:


model.intercept_


# In[10]:


model.predict([[5,3,1,2]])


# In[11]:


y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm


# In[12]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')

plt.ylabel('Truth')


# In[ ]:




