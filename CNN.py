#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[111]:


train_x = np.load('./x_train.npy')
train_y = np.load('./y_train.npy')


# In[113]:


test_x = np.load('./x_test.npy')
test_y = np.load('./y_test.npy')


# In[114]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.utils.class_weight import compute_class_weight


# In[115]:


class_weights = compute_class_weight(class_weight = "balanced" , classes=np.unique(train_y),  y=train_y)


# In[116]:


class_weights = dict(enumerate(class_weights))


# In[121]:


model = Sequential()      # initilaizing the Sequential nature for CNN model
# Adding the embedding layer which will take in maximum of 450 words as input and provide a 32 dimensional output of those words which belong in the top_words dictionary
model.add(Embedding(input_dim=6237, output_dim=128, input_length=67))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[122]:


epochs = 5
batch_size = 32


history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.1, class_weight=class_weights)


# In[124]:


from sklearn.metrics import classification_report

y_pred=model.predict(test_x, batch_size=256, verbose=1)


# In[125]:


print(classification_report(test_y, y_pred.round(), digits=4))

