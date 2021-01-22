#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd

df = pd.read_csv('total.csv') # read file


# In[11]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# # Text

# In[12]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, Add
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test = train_test_split(df, test_size=0.1)


# # Tokenization

# In[14]:


from konlpy.tag import Mecab

mecab = Mecab()

X_train['tokenized'] = X_train['text'].apply(mecab.morphs)
X_test['tokenized'] = X_test['text'].apply(mecab.morphs)


# In[15]:


x_train = X_train['tokenized'].values
y_train = X_train['label'].values
x_test= X_test['tokenized'].values
y_test = X_test['label'].values


# # encoding

# In[16]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)


# In[17]:


threshold = 2
total_cnt = len(tokenizer.word_index) 
rare_cnt = 0 
total_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value


# In[18]:


vocab_size = total_cnt - rare_cnt + 2


# In[19]:


tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)


# In[21]:


max_len = 67


# In[22]:


x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)


# In[ ]:


np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)


# # BiLSTM

# In[23]:


import re
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight


# In[24]:


class_weights = compute_class_weight(class_weight = "balanced" , classes=np.unique(y_train),  y=y_train)


# In[25]:


class_weights = dict(enumerate(class_weights))


# In[26]:


model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))


# In[27]:


model.summary()


# In[28]:


x_test.shape


# In[29]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# In[30]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5, callbacks=[es, mc], batch_size=256, validation_split=0.1, class_weight=class_weights)


# In[31]:


from sklearn.metrics import classification_report

y_pred=model.predict(x_test, batch_size=256, verbose=1)
report = classification_report(y_test, y_pred.round(), digits=4)
print(report)

