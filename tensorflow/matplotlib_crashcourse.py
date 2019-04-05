
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[21]:


#in non-Jupyter environmenr use plt.show()


# In[11]:


x= np.arange(0,10)
y= x**2


# In[15]:


x


# In[16]:


y


# In[39]:


plt.plot(x,y,'g--')
plt.xlim(0,4)
plt.ylim(0,10)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Matplit Chart')
#or can be with various characters or colours or styles


# In[42]:


mat = np.arange(0,100).reshape(10,10)


# In[44]:


plt.imshow(mat)


# In[52]:


plt.imshow(mat,cmap = 'inferno')


# In[57]:


map = np.random.randint(0,1000,(10,10))


# In[60]:


plt.imshow(map)
plt.colorbar()


# In[61]:


df = pd.read_csv ('salaries.csv')


# In[64]:


df.plot(x='Salary', y= 'Age', kind="scatter")


# In[ ]:




