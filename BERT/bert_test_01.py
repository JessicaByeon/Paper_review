# 네이버 영화 리뷰 corpus의 감성 이진분류(긍정/부정)


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F


train_df = pd.read_csv('D:\Paper_review\BERT\\nsmc\\nsmc/ratings_train.txt', sep='\t')
test_df = pd.read_csv('D:\Paper_review\BERT\\nsmc\\nsmc/ratings_test.txt', sep='\t')

print(train_df.info())

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# 전체 데이터의 40%만 사용
train_df = train_df.sample(frac=0.4, random_state=999)
test_df = test_df.sample(frac=0.4, random_state=999)

# print(train_df) # [59998 rows x 3 columns]
# print(test_df) # [19999 rows x 3 columns]

# print(train_df.info())
# <class 'pandas.core.frame.DataFrame'>      
# Int64Index: 59998 entries, 103096 to 149841
# Data columns (total 3 columns):
#  #   Column    Non-Null Count  Dtype       
# ---  ------    --------------  -----       
#  0   id        59998 non-null  int64       
#  1   document  59998 non-null  object      
#  2   label     59998 non-null  int64       
# dtypes: int64(2), object(1)



class NsmcDataset(Dataset):
    ### Naver Sentiment Movie Corpus Dataset ### 네이버 영화 리뷰 corpus의 감성 이진분류(긍정/부정)
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label

# train에 사용될 DataLoader를 만들어준다.
nsmc_train_dataset = NsmcDataset(train_df)
train_loader = DataLoader(nsmc_train_dataset, batch_size=3, shuffle=True, num_workers=0)

# Huggingface에서 구현된 Bert는 pytorch의 module클래스를 상속받고 있음
# 따라서 이미지 분류 classification task처럼 진행가능

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# google pre-trained model 사용
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') # 토큰에 대한 아이디 값을 리스트로 리턴

# 'BertForSequenceClassification' 모델 사용 -- 디폴트가 이진분류
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.to(device)


# 훈련 전 세팅
optimizer = Adam(model.parameters(), lr=1e-6)

itr = 1
p_itr = 500
epochs = 1
total_loss = 0
total_len = 0
total_correct = 0


import time
start = time.time()

# 훈련시 주의 : 학습 샘플의 인풋이 (batch_size, sequence_length)로 들어감
# 따라서 zero-padding을 직접 해줘서 model의 forward에 넣어줘야함

model.train()
for epoch in range(epochs):
    
    for text, label in train_loader:
        optimizer.zero_grad()
        
        # encoding and zero padding
        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
        
        sample = torch.tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
        labels = torch.tensor(label)
        outputs = model(sample, labels=labels)
        loss, logits = outputs

        pred = torch.argmax(F.softmax(logits), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if itr % p_itr == 0:
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, total_correct/total_len))
            total_loss = 0
            total_len = 0
            total_correct = 0

        itr+=1


# evaluation
model.eval()

nsmc_eval_dataset = NsmcDataset(test_df)
eval_loader = DataLoader(nsmc_eval_dataset, batch_size=3, shuffle=False, num_workers=0)

total_loss = 0
total_len = 0
total_correct = 0

for text, label in eval_loader:
    encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
    padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
    sample = torch.tensor(padded_list)
    sample, label = sample.to(device), label.to(device)
    labels = torch.tensor(label)
    outputs = model(sample, labels=labels)
    _, logits = outputs

    pred = torch.argmax(F.softmax(logits), dim=1)
    correct = pred.eq(labels)
    total_correct += correct.sum().item()
    total_len += len(labels)

print('Test accuracy: ', total_correct / total_len)

end = time.time()
print("걸린시간 : ", f"{end - start: .5f} sec")


