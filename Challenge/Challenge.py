#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# Download the dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


# Use a small dataset
train_size = 300
test_size = 50


# In[4]:


x_train = x_train[:train_size]
y_train = y_train[:train_size]
x_test = x_test[:test_size]
y_test = y_test[:test_size]


# In[5]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[12]:


# Download the pre-trained model
from keras.applications.mobilenet import MobileNet
MobileNet_extractor = MobileNet(weights='imagenet', include_top=False)


# In[ ]:


# (1) Reshape inputs into a size MobileNet requires
# (2) Extract features
# (3) Build a model


# In[17]:


from keras.models import Sequential
from keras.layers import Dense, Activation


# In[18]:


model = Sequential()
# Build a simple model
# Input of this model is the output of MobileNet_extractor
# Output of this model is your predictions
model.summary()


# In[19]:


model.compile(...)


# In[20]:


model.fit(...)

