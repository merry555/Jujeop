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



import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")



train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')

train.type = train.type-1
test.type = test.type-1



x_train = train['tokenized'].values
y_train = train['type'].values
x_test= test['tokenized'].values
y_test = test['type'].values



tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)


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




vocab_size = total_cnt - rare_cnt + 2


threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)



print('최대 길이 :',max(len(l) for l in x_train))
print('평균 길이 :',sum(map(len, x_train))/len(x_train))



tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)



max_len = 58



x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)



import re
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_class_weight





class_weights = compute_class_weight(class_weight = "balanced" , classes=np.unique(y_train),  y=y_train)
class_weights = dict(enumerate(class_weights))


# BILSTM MODEL

model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(4, activation='softmax'))


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_bilstm.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, callbacks=[es, mc], batch_size=16, validation_split=0.1, class_weight=class_weights)


from sklearn.metrics import classification_report

y_pred=model.predict(x_test, batch_size=16, verbose=1)


y_pred_ = np.asarray(y_pred)



y_pred_.argmax(axis=1)



print(classification_report(y_test, y_pred_.argmax(axis=1), digits=4))

# CNN MODEL

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.utils.class_weight import compute_class_weight


loss = tf.nn.softmax_cross_entropy_with_logits
loss = 'sparse_categorical_crossentropy'
model = Sequential()
model.add(Embedding(input_dim=6536, output_dim=128, input_length=58))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
model.summary()


epochs = 10
batch_size = 32

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, class_weight=class_weights)



from sklearn.metrics import classification_report

y_pred=model.predict(x_test, batch_size=32, verbose=1)


y_pred_ = np.asarray(y_pred)
print(classification_report(y_test, y_pred_.argmax(axis=1),digits=4))

