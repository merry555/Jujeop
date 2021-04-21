#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')


# ## BiLSTM

# In[9]:


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


# In[ ]:


from konlpy.tag import Mecab

mecab = Mecab()

train['tokenized'] = train['text'].apply(mecab.morphs)
test['tokenized'] = test['text'].apply(mecab.morphs)


# In[ ]:


x_train = train['tokenized'].values
y_train = train['label'].values
x_test= test['tokenized'].values
y_test = test['label'].values


# In[ ]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)


# In[ ]:


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


# In[ ]:


vocab_size = total_cnt - rare_cnt + 2


# In[ ]:


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

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)


# In[ ]:


print('최대 길이 :',max(len(l) for l in x_train))
print('평균 길이 :',sum(map(len, x_train))/len(x_train))


# In[ ]:


tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)


# In[ ]:


max_len = 58


# In[ ]:


x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)


# In[ ]:


import re
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight


# In[ ]:


model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# In[ ]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, callbacks=[es, mc], batch_size=16, validation_split=0.1)


# In[ ]:


from sklearn.metrics import classification_report

y_pred=model.predict(x_test, batch_size=16, verbose=1)
report = classification_report(y_test, y_pred.round(), digits=4)
print(report)


# ## CNN

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.utils.class_weight import compute_class_weight


# In[ ]:


model = Sequential()
model.add(Embedding(input_dim=6536, output_dim=128, input_length=58))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


epochs = 10
batch_size = 32


history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)


# In[ ]:


from sklearn.metrics import classification_report

y_pred=model.predict(x_test, batch_size=32, verbose=1)
report = classification_report(y_test, y_pred.round(), digits=4)
print(report)
