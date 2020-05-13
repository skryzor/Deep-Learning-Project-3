import pandas as pd
train_datapath = "/snli_1.0/snli_1.0/snli_1.0_train.jsonl"
df = pd.read_json(train_datapath)
print(df.head)