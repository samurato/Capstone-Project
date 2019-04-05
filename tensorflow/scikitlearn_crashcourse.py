#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[53]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split


# In[19]:


data = np.random.randint(0,100,(10,2))


# In[20]:


data


# In[21]:


scaler_model=MinMaxScaler()


# In[22]:


type(scaler_model)


# In[27]:


#Warning is ok!
scaler_model.fit(data)


# In[24]:


#Normalise data Values/maxvalue
scaler_model.transform(data)


# In[36]:


#Perform previous functions in one go.
result = scaler_model.fit_transform(data)


# In[37]:


result


# In[47]:


data = pd.DataFrame(data =np.random.randint(0,101,(50,4)),columns = ['f1','f2','f3','label'])
#data = pd.DataFrame(data=np.random.randint(0,101,(50,4)),columns=['f1','f2','f3','label'])


# In[48]:


data


# In[59]:


X= data[['f1','f2','f3']]


# In[52]:


y= data[['label']]


# In[60]:


#Shift + Tab will open the documentation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[62]:


X_train.shape


# In[ ]:




