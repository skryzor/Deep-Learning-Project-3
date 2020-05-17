# -*- coding: utf-8 -*-
# First lets improve libraries that we are going to be used in this lab session
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
random.seed(134)
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 32

import zipfile
with zipfile.ZipFile("snli_1.0.zip", 'r') as zip_ref:
    zip_ref.extractall("snli_1.0")

train_datapath = "./snli_1.0/snli_1.0/snli_1.0_train.jsonl"
valid_datapath = "./snli_1.0/snli_1.0/snli_1.0_dev.jsonl"
test_datapath = "./snli_1.0/snli_1.0/snli_1.0_test.jsonl"

df_train = pd.read_json(train_datapath, lines = "true")
df_val = pd.read_json(valid_datapath, lines = "true")
# df3 = pd.read_json(test_datapath, lines = "true")
df_train['sentence_mix'] = df_train['sentence1'] + " " + df_train['sentence2']
df_train.loc[df_train.sentence_mix.str.len() > 0, "sentence_mix"] = df_train.sentence_mix.str.split(" ")
print(df_train.head())

df_val['sentence_mix'] = df_val['sentence1'] + " " + df_val['sentence2']
df_val.loc[df_val.sentence_mix.str.len() > 0, "sentence_mix"] = df_val.sentence_mix.str.split(" ")
print(df_val.head())

train_list = df_train['sentence_mix'].tolist()
val_list = df_val['sentence_mix'].tolist()
train_list = train_list + val_list

all_tokens = [i for j in train_list for i in j]
print(len(train_list), len(all_tokens))

def build_vocab(all_tokens):
    token_counter = Counter(all_tokens)
    vocab = token_counter.keys()
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token, vocab
token2id, id2token, vocab = build_vocab(all_tokens)
print(len(token2id), len(id2token), len(vocab))
# print(token2id, id2token)

# ordered_words = ['<pad>','<unk>'] + list(vocab) 
# words
def apply_idz(x):
    temp = []
    for a in x:
        if (token2id[a]):
            temp.append(token2id[a])
        else:
            temp.append(1)
    return temp

def idize(df):
    df['sentence_idz'] = df["sentence_mix"].apply(apply_idz)
    return df    

train_train = idize(df_train)
val_val = idize(df_val)
# print(train_train.columns)
train_train = train_train.drop(['annotator_labels', 'captionID', 'pairID', 'sentence1',
       'sentence1_binary_parse', 'sentence1_parse', 'sentence2',
       'sentence2_binary_parse', 'sentence2_parse', 'sentence_mix' ], axis=1)    
print(train_train.head)

val_val = val_val.drop(['annotator_labels', 'captionID', 'pairID', 'sentence1',
       'sentence1_binary_parse', 'sentence1_parse', 'sentence2',
       'sentence2_binary_parse', 'sentence2_parse', 'sentence_mix' ], axis=1)    
print(val_val.head)

def encode_target(train_train):
    train_train['gold_label'][train_train['gold_label']=='neutral']=0
    train_train['gold_label'][train_train['gold_label']=='entailment']=1
    train_train['gold_label'][train_train['gold_label']=='contradiction']=2
    return train_train

train_train = encode_target(train_train)
val_val = encode_target(val_val)
#val_val = encode_target(val_val)
print(train_train.head())
print(val_val.head())

class SnliDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s1 = self.df.iloc[idx]['sentence_idz']
        len1 = len(s1)
        tar = self.df.iloc[idx]['gold_label']
        if(tar != 0 and tar != 1 and tar != 2):
            tar = 0
        return [np.array(s1),np.array(len1),np.array(tar)]

MAX_LEN = 25

def vocab_collate_func(batch):
    data_list_s1 = []
    label_list = []
    length_list_s1 = []

    for datum in batch:
        label_list.append(datum[2])
        length_list_s1.append(datum[1])
    # padding
    for datum in batch:
        if datum[1]>MAX_LEN:
            padded_vec_s1 = np.array(datum[0])[:MAX_LEN]
        else:
            padded_vec_s1 = np.pad(np.array(datum[0]),
                                pad_width=((0,MAX_LEN - datum[1])),
                                mode="constant", constant_values=0)
        data_list_s1.append(padded_vec_s1)
    sentence = torch.from_numpy(np.array(data_list_s1))
    length = torch.LongTensor(np.array(length_list_s1))
    label = torch.LongTensor(np.array(label_list))
    return [sentence, length, label]


train_dataset = SnliDataset(train_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True)

val_dataset = SnliDataset(val_val)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True)

dataloaders = [train_loader, val_loader]
print(dataloaders[0])

class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size):
        super(RNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        # self.embedding.from_pretrained(torch.from_numpy(loaded_embeddings).cuda(), freeze = True)


        
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers, batch_first = True)
        self.linear1 = nn.Linear(hidden_size, 500)
        self.linear2 = nn.Linear(500,num_classes)

    def init_hidden(self, batch_size):
        hidden = torch.randn(self.num_layers, batch_size, self.hidden_size)
        return hidden.cuda()

    def forward(self, data, length):  
        batch_size = data.size(0)
        self.hidden1 = self.init_hidden(batch_size)

        embed1 = self.embedding(data)
        
        # print(embed1.size())
        rnn_out, hidden1 = self.rnn(embed1, self.hidden1)
        # print(self.hidden1.size())

        rnn_out1 = torch.sum(hidden1, dim=0)
        # print(rnn_out1.size())
        # print(combined_out.size())
        logits = F.relu(self.linear1(rnn_out1))
        res = self.linear2(logits)
        
        return res

def training(model,criterion, optimizer, name, num_epochs):
    best_loss = np.inf
    best_acc = 0
    loss_hist = {'train':[],'validate':[]}
    acc_hist = {'train':[],'validate':[]}
    for i in range(num_epochs):
        for enu,phase in enumerate(['train', 'validate']):
            print(enu, phase)
            running_loss = 0
            running_total = 0
            correct = 0
            total = 0
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
            for (data, length, labels) in dataloaders[enu]:
#                 data_batch1, length_batch1, data_batch2, len_batch2, label_batch = data1, length1, data2, length2, labels
                data_batch, len_batch, label_batch = data.cuda(), length.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(data_batch, len_batch)
                loss = criterion(outputs, label_batch)
                if phase=='train':
                    loss.backward()
                    optimizer.step()
                N = labels.size(0)
                
                outputs = F.softmax(outputs, dim=1)
                predicted = outputs.max(1, keepdim=True)[1]
#                 print(type(predicted))
                total += labels.size(0)
                correct += predicted.eq(label_batch.view_as(predicted)).sum().item()
                running_loss += loss.data * N
                running_total += N
            epoch_loss = running_loss/running_total
            loss_hist[phase].append(epoch_loss.item())
            accuracy = 100 * correct / total
            acc_hist[phase].append(accuracy)
            print('Epoch: {}, Phase: {}, epoch loss: {:.4f}, accuracy: {:.4f}'\
                      .format(i,phase,epoch_loss, accuracy))
        if phase == 'validate' and best_acc < accuracy:
            best_loss = epoch_loss
            best_acc = accuracy
            torch.save(model,name)
    print('Best val dice loss: {:4f}, Best Accuracy: {:4f}'.format(best_loss,best_acc))
    return model, loss_hist, acc_hist


model = RNN(300,500,1,3,100000+2).cuda()
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
m_save, loss_hists, acc_hist = training(model,criterion,optimizer,"model_rnn",15)
print(loss_hists, acc_hist)

plt.plot(loss_hists['train'],label="train loss")
plt.plot(loss_hists['validate'],label="val loss")
plt.legend()
plt.savefig("val_loss.png")

plt.plot(acc_hist['train'],label="train acc")
plt.plot(acc_hist['validate'],label="val acc")
plt.legend()
plt.savefig("val_accuracy.png")
