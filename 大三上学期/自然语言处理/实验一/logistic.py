from multiprocessing.connection import wait
import re
import string
from tkinter import Label
from turtle import pd
import numpy as np
import os
import jieba
import pandas as pd
import math
import random


# 定义超参数
epoch = 5
batch_size = 1
# batch_size = 100
learning_rate = 0.0001
# learning_rate = 0.01

# sigmoid 激活函数
def sigmoid(x): 
    return 1/(1+np.exp(-x))

def dsigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

# # ReLu 激活函数
# def ReLu(x):
#     x = (np.abs(x) + x) / 2.0
#     return x

# def dReLu(x):
#     if (x <= 0):
#          return 0
#     else:
#         return 1

# # 线性激活函数
# def Linear(x):
#     return x

# def dLinear(x):
#     return 1


def get_dir(path):
    file_list_name = os.listdir(path)
    file_list_address = [os.path.join(path,x) for x in file_list_name]
    return file_list_address


def analysis_content(content):
    content = re.sub(r'[0-9]', '', content)
    content = re.sub(r'https?:\/\/.*[\r\n]*', '', content)
    word_list_clearn = []
    word_list = jieba.lcut(sentence = content)
    for word in word_list:
        if(word not in string.punctuation and word != ' ' and word not in Stopword):
            word_list_clearn.append(word)
    return word_list_clearn


def Partition_data(file_list_address):
    Train_file_list_address = random.sample(file_list_address, math.ceil(len(file_list_address)*0.8))
    Test_file_list_address =  [i for i in file_list_address if i not in Train_file_list_address]
    return Train_file_list_address , Test_file_list_address


def analysis_file(file_address):
    content = ''
    if (os.path.exists(file_address)):
        with open(file_address,'r',encoding='utf-8',errors='ignore') as f:
            for line in f.readlines():
                line = line.strip().replace(u'\u3000',u'')
                content = content + line
    return content

def Parse_file(Path):
    Dic = {}
    with open(Path, 'r', encoding='utf-8')as f:
        for i, line in enumerate (f.readlines()):
            line = line.strip().replace(u'\u3000',u'')
            if i == 0:
                continue
            word_msg = line.split()
            word_name = word_msg[0]
            word_vec = word_msg[1:]
            word_vec = np.array(list(map(float, word_vec)))
            Dic[word_name] = word_vec
    return Dic

Dic = Parse_file("C:\\Users\\16176\\Desktop\\sgns.baidubaike.bigram-char")

def get_data(corpus_address, flag):
    data = []
    for address in corpus_address:
        content = analysis_file(address)
        word_list = analysis_content(content)
        data.append((word_list, flag))
    return data


def Partition_batch(data):
    Data_batch = []
    batch_data = []

    for i in data:
        batch_data.append(i)
        if len(batch_data) == batch_size:
            Data_batch.append(batch_data)
            batch_data = []

    if len(batch_data) != batch_size and len(batch_data) != 0:
        Data_batch.append(batch_data)

    return Data_batch


def find_vec(word, Dic):
    if word in Dic.keys():
        return Dic[word]
    else:
        return np.array([0.]*300)


Stopword = [line.strip() for line in open('C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data\\vocabulary\\stopwords_utf8.txt', 'r',encoding='utf-8').readlines()]
Corpus_address = "C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data"
Pos_corpus_address = get_dir(Corpus_address + '\\positive')
Train_pos_corpus_address , Test_pos_corpus_address = Partition_data(Pos_corpus_address)
Train_pos_data = get_data(Train_pos_corpus_address, 1)
Train_pos_data_batch = Partition_batch(Train_pos_data)
Test_pos_data = get_data(Test_pos_corpus_address, 1)


Neg_corpus_address = get_dir(Corpus_address + '\\negative')
Train_neg_corpus_address , Test_neg_corpus_address = Partition_data(Neg_corpus_address)
Train_neg_data = get_data(Train_neg_corpus_address, 0)
Train_neg_data_batch = Partition_batch(Train_neg_data)
Test_neg_data = get_data(Test_neg_corpus_address, 0)


Train_data_batch = Train_pos_data_batch + Train_neg_data_batch
Test_data = Test_pos_data + Test_neg_data

random.shuffle(Train_data_batch)

# w = np.array([1]*300)
w = np.array([0.01]*300)
# w = np.random.random(300)
# b = 10
b = 0.


for i in range(epoch):
    
    # batch_training
    for i,batch_data in enumerate(Train_data_batch):
        w_grad = np.array([0.]*300)
        b_grad = 0
        for data in batch_data:
            word_list = data[0]
            flag = data[1]
            word_vec = np.array([0.]*300)
            for word in word_list:
                word_vec += find_vec(word, Dic)

            y = np.dot(word_vec, w) + b
            
            w_grad += 2 * (sigmoid(y) - flag) * dsigmoid(y) * word_vec # array * array 对应元素相乘
            b_grad += 2 * (sigmoid(y) - flag) * dsigmoid(y)
            # print(sigmoid(y), dsigmoid(y), 2 * (sigmoid(y) - flag) * dsigmoid(y) , word_vec, 2 * (sigmoid(y) - flag) * dsigmoid(y) * word_vec)
        
        w_grad = w_grad/len(batch_data)
        b_grad = b_grad/len(batch_data)
        w = w - w_grad * learning_rate
        b = b - b_grad * learning_rate

# print(w)
# print(b)

# 测试二分类器
score = 0
Interference = 0
for data in Test_data:
    word_list = data[0]
    flag = data[1]
    word_vec = np.array([0.]*300)
    for word in word_list:
        word_vec += find_vec(word, Dic)
    y = sigmoid(np.dot(word_vec, w) + b)
    if y > 0.5:
        if flag == 1:
            score += 1

    if y < 0.5:
        if flag == 0:
            score += 1

    if y == 0.5:
        Interference += 1

score = score/(len(Test_data) - Interference)
print('基于逻辑回归的情感分析系统准确率为',score)


# 痛定思痛，决定调用库函数
# 调用 SKlearn 库
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 解析词向量文件
# def Parse_file(Path):
#     Dic = {}
#     with open(Path, 'r', encoding='utf-8')as f:
#         for i, line in enumerate (f.readlines()):
#             line = line.strip().replace(u'\u3000',u'')
#             if i == 0:
#                 continue
#             word_msg = line.split()
#             word_name = word_msg[0]
#             word_vec = word_msg[1:]
#             word_vec = np.array(list(map(float, word_vec)))
#             Dic[word_name] = word_vec
#     return Dic


# # 读取字典
# Dic = Parse_file("C:\\Users\\16176\\Desktop\\sgns.baidubaike.bigram-char")

# def find_vec(word):
#     if word in Dic.keys():
#         return Dic[word]
#     else:
#         return np.array([0.]*300)


# def get_dir(path):
#     file_list_name = os.listdir(path)
#     file_list_address = [os.path.join(path,x) for x in file_list_name]
#     return file_list_address

# def analysis_file(file_address):
#     content = ''
#     if (os.path.exists(file_address)):
#         with open(file_address,'r',encoding='utf-8',errors='ignore') as f:
#             for line in f.readlines():
#                 line = line.strip().replace(u'\u3000',u'')
#                 content = content + line
#     return content

# def analysis_content(content):
#     # 利用正则表达式筛选掉一部分与情感分析无关的word
#     content = re.sub(r'[0-9]*', '', content)
#     content = re.sub(r'https?:\/\/.*[\r\n]*', '', content)
#     word_list_clearn = []
#     word_list = jieba.lcut(sentence = content)
#     for word in word_list:
#         if(word not in string.punctuation and word != ' ' and word not in Stopword):
#             word_list_clearn.append(word)
#     word_vec = np.array([0.] * 300)
#     for word in  word_list_clearn:
#         word_vec =  np.add(word_vec, find_vec(word))
#     return word_vec

# def get_data(corpus_address):
#     data = []
#     for address in corpus_address:
#         content = analysis_file(address)
#         word_array = analysis_content(content)
#         data.append(word_array)
#     return np.array(data)

    
# Stopword = [line.strip() for line in open('C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data\\vocabulary\\stopwords_utf8.txt', 'r',encoding='utf-8').readlines()]
# Corpus_address = "C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data"
# Pos_corpus_address = get_dir(Corpus_address + '\\positive')
# Pos_data = get_data(Pos_corpus_address)
# Neg_corpus_address = get_dir(Corpus_address + '\\negative')
# Neg_data = get_data(Neg_corpus_address)
# Data = np.append(Pos_data, Neg_data, axis=0)
# Data_label = np.append(np.array([1.]*1000), np.array([0.]*1000))  

# # 以 80% - 20% 比例将数据集分为训练集和测试集
# Data_train, Data_test, Data_label_train, Data_label_test = train_test_split(
#     Data, Data_label,
#     train_size=0.8, test_size=0.2, random_state=188
# )

# # 初始化模型
# clf = LogisticRegression(
#     penalty="l2", C=1.0, random_state=None, solver="sag", max_iter=3000,
#     multi_class='ovr'
# )

# # 训练模型
# clf.fit(Data_train, Data_label_train)

# #输出参数
# print(clf.intercept_, clf.coef_)

# # 测试模型
# Data_label_pred = clf.predict(Data_test)

# # 输出准确率
# print(roc_auc_score(Data_label_test, Data_label_pred))



