print("Kande Nilesh Abhimanyu")
print("Sr. No. - 15688")
print("CSA, IISc, Bangalore")
print("Deep Learning Project III")

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
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV


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
df_test = pd.read_json(test_datapath, lines = "true")


###################################TFIDF############################################
print("calculating for tfidf")
test_label = np.asarray(df_test['gold_label'].tolist())
s1 = df_test['sentence1'].tolist()
s2 = df_test['sentence2'].tolist()
test_list = []
for i in range(len(s1)):
  test_list.append(s1[i] + s2[i])
test_list = np.asarray(test_list)

train_label = np.asarray(df_train['gold_label'].tolist())
s1 = df_train['sentence1'].tolist()
s2 = df_train['sentence2'].tolist()
train_list = []
for i in range(len(s1)):
  train_list.append(s1[i] + s2[i])
train_list = np.asarray(train_list)

tfidf = TfidfVectorizer(max_features=512)
model1 = LogisticRegressionCV(verbose=1, max_iter=128)
x = [text for text in train_list]
y = [text for text in train_label]
x_transformed = tfidf.fit_transform(x)

x_test = [text for text in test_list]
y_test = [text for text in test_label]
x_test_transformed = tfidf.transform(x_test)

filename = 'tfidf_model'
model1 = pickle.load(open(filename, 'rb'))
pred = model1.predict(x_test_transformed)
with open("tfidf.txt", 'w') as f:
	for i in pred:
		f.write(i + "\n")
test_score = model1.score(x_test_transformed, y_test)
print("test_accuracy for tfidf model " + str(test_score))
print("calculated for tfidf")
######################################RNN##########################################

print("calculating for RNN")

# df3 = pd.read_json(test_datapath, lines = "true")
df_train['sentence_mix'] = df_train['sentence1'] + " " + df_train['sentence2']
df_train.loc[df_train.sentence_mix.str.len() > 0, "sentence_mix"] = df_train.sentence_mix.str.split(" ")
# print(df_train.head())

df_val['sentence_mix'] = df_val['sentence1'] + " " + df_val['sentence2']
df_val.loc[df_val.sentence_mix.str.len() > 0, "sentence_mix"] = df_val.sentence_mix.str.split(" ")
# print(df_val.head())

df_test['sentence_mix'] = df_test['sentence1'] + " " + df_test['sentence2']
df_test.loc[df_test.sentence_mix.str.len() > 0, "sentence_mix"] = df_test.sentence_mix.str.split(" ")
# print(df_test.head())


train_list = df_train['sentence_mix'].tolist()
val_list = df_val['sentence_mix'].tolist()
test_list = df_val['sentence_mix'].tolist()

train_list = train_list + val_list + test_list
all_tokens = [i for j in train_list for i in j]
# print(len(train_list), len(all_tokens))

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
# print(len(token2id), len(id2token), len(vocab))

def apply_idz(x):
    temp = []
    for a in x:
        if a in token2id:
            temp.append(token2id[a])
        else:
            temp.append(1)
    return temp

def idize(df):
    df['sentence_idz'] = df["sentence_mix"].apply(apply_idz)
    return df    

test_test = idize(df_test)

test_test = test_test.drop(['annotator_labels', 'captionID', 'pairID', 'sentence1',
       'sentence1_binary_parse', 'sentence1_parse', 'sentence2',
       'sentence2_binary_parse', 'sentence2_parse', 'sentence_mix' ], axis=1)    
# print(test_test.head)

def encode_target(test_test):
    test_test['gold_label'][test_test['gold_label']=='neutral']=0
    test_test['gold_label'][test_test['gold_label']=='entailment']=1
    test_test['gold_label'][test_test['gold_label']=='contradiction']=2
    return test_test

test_test = encode_target(test_test)
# print(test_test.head())

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


test_dataset = SnliDataset(test_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=False)

class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size):
        super(RNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)


        
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
        
        rnn_out, hidden1 = self.rnn(embed1, self.hidden1)

        rnn_out1 = torch.sum(hidden1, dim=0)
        logits = F.relu(self.linear1(rnn_out1))
        res = self.linear2(logits)
        
        return res


model = RNN(300,500,1,3,100000+2).cuda()
model = torch.load('model_rnn')
model.eval()
model.train(False)

phase = 'test'
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)


with open("deep_model.txt", 'w') as f:
	running_loss = 0
	running_total = 0
	correct = 0
	total = 0
	prediction = []
	for (data, length, labels) in test_loader:
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
		for i in predicted.cpu().data.numpy():
			if i == 0:
				f.write("neutral\n")
			elif i == 1:
				f.write("entailment\n")
			else:
				f.write("contradiction\n")
		total += labels.size(0)
		correct += predicted.eq(label_batch.view_as(predicted)).sum().item()
		running_loss += loss.data * N
		running_total += N

	epoch_loss = running_loss/running_total
	accuracy = 100 * correct / total
	print('Phase: {}, test loss: {:.4f}, accuracy: {:.4f}'\
	          .format(phase,epoch_loss, accuracy))

print("calculated for RNN model")