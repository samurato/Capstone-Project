
# coding: utf-8

# In[1]:


import numpy as np;


# In[2]:


my_list= [1,2,3];


# In[3]:


my_list


# In[5]:


type(my_list)


# In[7]:


arr = np.array(my_list);


# In[8]:


arr


# In[9]:


type(arr)


# In[14]:


np.arange(0,10)


# In[15]:


np.arange(0,10,2)


# In[16]:


np.zeros(5)


# In[18]:


np.zeros((3,5))


# In[19]:


np.ones((3,7))


# In[20]:


np.linspace(0,10)


# In[25]:


np.random.randint(0,20)


# # random integer generate with 3 by 3 array
# np.random.randint(0,100,(3,3))

# In[30]:


np.random.seed(101)
np.random.randint(0,100,(1,20))


# In[37]:


np.random.seed(101)
arr = np.random.randint(0,100,(1,20))


# In[38]:


arr


# In[45]:


arr.argmax()


# In[46]:


arr.argmin()


# In[53]:


arr.reshape(2,10)


# # Generate and arrange value from 0 to 100 and reshape to 10 by 10 matrix
# mat= np.arange(0,100).reshape(10,10)

# In[56]:


mat


# mat[7,5]

# # extract the value of 7th row 5th coloumn
# mat[7,5]

# # extract the value of all row in 0th coloumn
# mat[:,0]

# In[68]:


mat[0,:]


# # Extract 0 to 3 and 0 to 3 
# 
# mat[0:3,0:3]

# In[61]:


mat[6:9, 6:9]


# In[72]:


# Show arrays that is holding are greater than 50

mat>50


# In[73]:


#Filter all the matrices greater than 50
mat[mat>50]


# 
