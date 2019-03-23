
# coding: utf-8

# In[1]:


pwd


# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('salaries.csv')


# In[4]:


df


# In[5]:


df['Salary']


# In[7]:


df[['Name','Salary']]


# In[9]:


df.describe()


# In[10]:


df['Salary']>50


# In[13]:


df[df['Salary']>50000]


# In[ ]:




