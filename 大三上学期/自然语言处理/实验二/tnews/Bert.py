from transformers import BertModel , BertTokenizer , BertConfig
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os
import json
from torch.utils.data import  DataLoader

BERT_PATH = "bert-base-chinese"


class BertClassifier(nn.Module):
    def __init__(self, dropout = 0.2):
        super(BertClassifier, self).__init__()
        self.bert_config = BertConfig.from_pretrained('bert-base-chinese')
        self.bert = BertModel.from_pretrained(BERT_PATH , config = self.bert_config)
        # self.lay1 = nn.Sequential(nn.Linear(768, 256),
        #                           nn.ReLU(),
        #                           nn.Dropout(dropout))
        # self.lay2 = nn.Sequential(nn.Linear(256, 128),
        #                           nn.ReLU(),
        #                           nn.Dropout(dropout))
        # self.lay3 = nn.Linear(128, 15)

        # 定义分类器
        self.classifier = nn.Linear(self.bert_config.hidden_size , 15)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        pooled_output = bert_output[1]
        # x = self.lay1(x)
        # x = self.lay2(x)
        # x = self.lay3(x)
        prob = self.classifier(pooled_output)
        return torch.softmax(prob, dim = 1)


def train(model , train_dataset , valid_dataset):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = lr)

    # 记录文件
    file = open('Bert_record_file.txt' , 'w', encoding='utf-8')

    # 训练
    for epoch in range(1 , epochs + 1):

        # 训练数据集的准确率 损失率
        losses = 0 
        accuracy = 0 

        model.train()
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        train_bar = tqdm(train_dataloader , ncols = 100)

        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            print(type(input_ids))
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)
            print(type(input_ids))
            # 输出
            output = model(input_ids = input_ids.to(device), attention_mask = attention_mask.to(device), token_type_ids = token_type_ids.to(device))

            # 计算损失
            print(label_id ,'111')
            print(output,'222')
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            # 计算精度
            pred_labels = torch.argmax(output, dim = 1) 
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)
            accuracy += acc

            # 模型更新
            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss = loss.item(), acc = acc)

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)
        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

        # 写入记录文件
        txt = '\tTrain ACC:' + str(average_acc) + '\tLoss:' + str(average_loss) + '\n'
        file.write(txt)

        model.eval()
        losses = 0
        accuracy = 0
        valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
        valid_bar = tqdm(valid_dataloader, ncols = 100)

        for input_ids, token_type_ids, attention_mask, label_id  in valid_bar:

            valid_bar.set_description('Epoch %i valid' % epoch)

            output = model(input_ids = input_ids.to(device), attention_mask = attention_mask.to(device), token_type_ids = token_type_ids.to(device))

            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            # 预测文本类别
            pred_labels = torch.argmax(output, dim = 1)
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)
            accuracy += acc
            valid_bar.set_postfix(loss=loss.item(), acc = acc)

        average_loss = losses / len(valid_dataloader)
        average_acc = accuracy / len(valid_dataloader)

        print('\tValid ACC:', average_acc, '\tLoss:', average_loss)

        # 写入记录文件
        txt = '\tValid ACC:' + str(average_acc) + '\tLoss:' + str(average_loss) + '\n'
        file.write(txt)
    
    # 关闭文件
    file.close()

# 数据处理
class Dictionary(object):
    def __init__(self, path):
        self.word2tkn = {}
        self.tkn2word = []
        self.label2idx = {}
        self.idx2label = []

        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary(path)
        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))

    def tokenize(self, path, test_mode = False):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        with open(path, 'r', encoding = 'utf8') as f:
            lines = f.readlines()
            if test_mode:
                dic = {}
                input_ids = []
                token_type_ids = []
                attention_mask = []

                for line in tqdm(lines, ncols=100):

                    one_data = json.loads(line)
                    text = one_data['sentence']

                    token = tokenizer(text, add_special_tokens = True, padding = 'max_length', truncation = True, max_length = 150)
                    input_ids.append(np.array(token['input_ids']))
                    token_type_ids.append(np.array(token['token_type_ids']))
                    attention_mask.append(np.array(token['attention_mask']))
                dic['input_ids'] = input_ids
                dic['token_type_ids'] = token_type_ids
                dic['attention_mask'] = attention_mask  

                return dic

            else:
                
                i = 0

                dic = {}
                input_ids = []
                token_type_ids = []
                attention_mask = []
                label_id = []

                for line in tqdm(lines, ncols=100):

                    if i > 100:
                        break

                    one_data = json.loads(line)
                    text = one_data['sentence']
                    label = one_data['label']

                    # bert 分词
                    token = tokenizer(text, add_special_tokens = True, padding = 'max_length', truncation = True, max_length = 150)
                    input_ids.append(np.array(token['input_ids']))
                    token_type_ids.append(np.array(token['token_type_ids']))
                    attention_mask.append(np.array(token['attention_mask']))
                    label_id.append(self.dictionary.label2idx[label])
            

                    i += 1

                dic['input_ids'] = input_ids
                dic['token_type_ids'] = token_type_ids
                dic['attention_mask'] = attention_mask
                dic['label_id'] = label_id

                return dic

class tNewsDataset(object):
    def __init__(self, dic):
        self.input_ids = dic['input_ids']
        self.token_type_ids = dic['token_type_ids']
        self.attention_mask = dic['attention_mask']
        self.label_id = dic['label_id']

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_id[index]

    def __len__(self):
        return len(self.input_ids)

# 超参数
batch_size = 1
epochs = 6
lr = 5e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_folder = 'data/tnews_public'
dataset = Corpus(dataset_folder)
train_dataset = tNewsDataset(dataset.train)
valid_dataset = tNewsDataset(dataset.valid)

model = BertClassifier().to(device)

train(model , train_dataset , valid_dataset)