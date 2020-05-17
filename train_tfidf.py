# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV
import pickle

import zipfile
with zipfile.ZipFile("snli_1.0.zip", 'r') as zip_ref:
    zip_ref.extractall("snli_1.0")

train_datapath = "./snli_1.0/snli_1.0/snli_1.0_train.jsonl"
valid_datapath = "./snli_1.0/snli_1.0/snli_1.0_dev.jsonl"
test_datapath = "./snli_1.0/snli_1.0/snli_1.0_test.jsonl"

df1 = pd.read_json(train_datapath, lines = "true")
df2 = pd.read_json(valid_datapath, lines = "true")
df3 = pd.read_json(test_datapath, lines = "true")
print(len(df1["sentence1"]))
print(len(df2["sentence1"]))
print(len(df3["sentence1"]))
s1 = df1['sentence1'].tolist()
s2 = df1['sentence2'].tolist()
train_list = []
for i in range(len(s1)):
  train_list.append(s1[i] + s2[i])

s1 = df3['sentence1'].tolist()
s2 = df3['sentence2'].tolist()
test_list = []
for i in range(len(s1)):
  test_list.append(s1[i] + s2[i])

train_label = np.asarray(df1['gold_label'].tolist())
test_label = np.asarray(df3['gold_label'].tolist())
train_list = np.asarray(train_list)
test_list = np.asarray(test_list)
print(len(train_list), len(test_list), len(train_label), len(test_label))

tfidf = TfidfVectorizer(max_features=512)
model = LogisticRegressionCV(verbose=1, max_iter=128)

x = [text for text in train_list]
y = [text for text in train_label]
x_transformed = tfidf.fit_transform(x)
model.fit(x_transformed, y)

x_test = [text for text in test_list]
y_test = [text for text in test_label]
x_test_transformed = tfidf.transform(x_test)

train_score = model.score(x_transformed, y)

filename = 'tfidf_model'
pickle.dump(model, open(filename, 'wb'))

test_score = model.score(x_test_transformed, y_test)

result_base = "Train Accuracy: {train_acc:<.1%}  Test Accuracy: {test_acc:<.1%}"
result = result_base.format(train_acc=train_score, test_acc=test_score)
print(result)

