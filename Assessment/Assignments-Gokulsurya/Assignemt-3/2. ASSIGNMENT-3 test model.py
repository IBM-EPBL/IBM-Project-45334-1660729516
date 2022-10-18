#!/usr/bin/env python
# coding: utf-8

# # test the model

# In[4]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[5]:


import numpy as np


# In[6]:


model=load_model("flowers.h5")


# In[7]:


img=image.load_img(r"D:\New folder\ibm\assignment 3\sunflower.jpg",target_size=(64,64))


# In[8]:


img


# In[9]:


type(img)


# In[10]:


x=image.img_to_array(img)


# In[11]:


x


# In[12]:


x.shape


# In[13]:


x=np.expand_dims(x,axis=0)


# In[14]:


x.shape


# In[15]:


pred_prob=model.predict(x)


# In[16]:


pred_prob


# In[17]:


pred_prob=model.predict(x)


# In[18]:


pred_id=pred_prob.argmax(axis=1)[0]


# In[19]:


pred_id


# In[20]:


print("predicted flower is",str(class_name[pred_id]))


# In[22]:


img=image.load_img(r"D:\New folder\ibm\assignment 3\rose.jpg",target_size=(64,64))


# In[23]:


img


# In[24]:


x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred_prob=model.predict(x)
pred_id=pred_prob.argmax(axis=1)[0]
print("predicted flower is",str(class_name[pred_id]))


# In[25]:


img=image.load_img(r"D:\New folder\ibm\assignment 3\tulip.jpg",target_size=(64,64))


# In[26]:


img


# In[27]:


x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred_prob=model.predict(x)
pred_id=pred_prob.argmax(axis=1)[0]
print("predicted flower is",str(class_name[pred_id]))


# In[ ]:




