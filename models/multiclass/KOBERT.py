
# coding: utf-8

# In[2]:


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule


# In[17]:


##GPU 
device = torch.device("cuda:1")


# In[18]:


bertmodel, vocab = get_pytorch_kobert_model()


# In[19]:


# Import all libs
import os
import numpy as np
import pandas as pd
import re
import csv
from collections import namedtuple
from sklearn.model_selection import train_test_split
import nltk
from sklearn.pipeline import Pipeline
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import sys


# In[20]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[21]:



train.type = train.type-1
test.type = test.type-1


# In[22]:


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


# In[23]:


max_len = 128
batch_size = 8
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  2e-5


# In[24]:


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


# In[25]:


dataset_train = [(d,l) for d,l in zip(train['text'], train['type'])]
dataset_test = [(d,l) for d,l in zip(test['text'], test['type'])]


# In[26]:


data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)


# In[27]:


data_train[0]


# In[28]:


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=4,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


# In[29]:


model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)


# In[30]:


# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


# In[31]:


optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


# In[32]:


t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)


# In[33]:


scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)


# In[34]:


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


# In[35]:


for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        print(valid_length)
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)

        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))


# In[36]:


S_LABELS = [0,1,2,3]

model.eval()

sentiment_true = []
sentiment_pred = []
for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)
      valid_length= valid_length
      label = label.long().to(device)
      out = model(token_ids, valid_length, segment_ids)
      max_val, max_indices = torch.max(out, 1)

      for idx in range(len(label)):
        sentiment_true.append(S_LABELS[int(label[idx])])
        sentiment_pred.append(S_LABELS[int(max_indices[idx])])
print(sentiment_true)
print(sentiment_pred)

from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

print('confusion matrix')
print(confusion_matrix(sentiment_true, sentiment_pred, labels=[0,1,2,3]))
print(classification_report(sentiment_true, sentiment_pred, digits=4))

from sklearn.metrics import f1_score
print(f1_score(sentiment_true, sentiment_pred, labels=None, pos_label=1, average='macro'))

