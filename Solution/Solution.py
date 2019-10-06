#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# Download the dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


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


# In[6]:


from skimage.transform import resize
x_train = resize(x_train, (x_train.shape[0], 224, 224), anti_aliasing=True)
x_test = resize(x_test, (x_test.shape[0], 224, 224), anti_aliasing=True)


# In[7]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[8]:


x_train = np.stack((x_train, x_train, x_train), axis=3)
x_test = np.stack((x_test, x_test, x_test), axis=3)


# In[9]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[10]:


from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[11]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[12]:


from keras.applications.mobilenet import MobileNet
MobileNet_extractor = MobileNet(weights='imagenet', include_top=False)


# In[13]:


train_features = MobileNet_extractor.predict(x_train)
test_features = MobileNet_extractor.predict(x_test)


# In[14]:


print(train_features.shape, y_train.shape)
print(test_features.shape, y_test.shape)


# In[15]:


train_features = train_features.reshape(train_features.shape[0],-1)
test_features = test_features.reshape(test_features.shape[0],-1)


# In[16]:


print(train_features.shape, y_train.shape)
print(test_features.shape, y_test.shape)


# In[17]:


from keras.models import Sequential
from keras.layers import Dense, Activation


# In[18]:


model = Sequential()
model.add(Dense(10, input_shape = (train_features.shape[1],), kernel_initializer='he_normal'))
model.add(Activation('softmax'))
model.summary()


# In[19]:


model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[20]:


model.fit(train_features, y_train, epochs=5, validation_data=(test_features, y_test))

