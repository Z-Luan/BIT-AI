import re
import string
from turtle import pd
import numpy as np
import os
import jieba
import pandas as pd
import math
import random

def get_dir(path):
    file_list_name = os.listdir(path)
    file_list_address = [os.path.join(path,x) for x in file_list_name]
    return file_list_address


def analysis_dir(file_list_address):
    content = ''
    for file_address in file_list_address:
        with open(file_address,'r',encoding='utf-8',errors='ignore') as f:
            for line in f.readlines():
                line = line.strip().replace(u'\u3000',u'')
                content = content + line
    return content


def analysis_content(content):
    content = re.sub(r'https?:\/\/.*[\r\n]*', '', content)
    content = re.sub(r'[0-9]', '', content)
    corpus_list = jieba.lcut(sentence = content)
    corpus_clearn_list = []
    # for word in corpus_list: 
    #     if(word in Stopword or word in string.punctuation or word == ' ' or word in Degree_pd.word):
    #         corpus_list.remove(word)
    for word in corpus_list: 
        if(word not in Stopword and word not in string.punctuation and word != ' '):
            corpus_clearn_list.append(word)
    return corpus_clearn_list


def Partition(file_list_address):
    Train_file_list_address = random.sample(file_list_address, math.ceil(len(file_list_address)*0.8))
    Test_file_list_address =  [i for i in file_list_address if i not in Train_file_list_address]
    return Train_file_list_address , Test_file_list_address


def Transform(List):
    lengh = len(List)
    Dic = {}
    test = 0
    for word in List:
        Dic[word] = Dic.get(word, 0) + 1
    for word in Dic.keys():
        Dic[word] = Dic[word]/lengh
        test += Dic[word]
    # print(test)
    
    return Dic


def analysis_file(file_address):
    content = ''
    if (os.path.exists(file_address)):
        with open(file_address,'r',encoding='utf-8',errors='ignore') as f:
            for line in f.readlines():
                line = line.strip().replace(u'\u3000',u'')
                content = content + line
    return content

def write_txt(Object, Path):
    with open ("C:\\Users\\16176\\Desktop\\test" + Path + '.txt', mode='a') as f:
        if (type(Object).__name__ == 'dict'):
            for word in Object.items():
                f.write(str(word) + '\n')
        if (type(Object).__name__ == 'list'):
            for word in Object:
                f.write(str(word) + '\n')

Stopword = [line.strip() for line in open('C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data\\vocabulary\\stopwords_utf8.txt', 'r',encoding='utf-8').readlines()]
Corpus_address = "C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data"
Pos_corpus_address = get_dir(Corpus_address + '\\positive')
Train_pos_corpus_address , Test_pos_corpus_address = Partition(Pos_corpus_address)
Train_pos_corpus_content = analysis_dir(Train_pos_corpus_address)
Train_pos_corpus_list = analysis_content(Train_pos_corpus_content)
Pos_dic = Transform(Train_pos_corpus_list)

Neg_corpus_address = get_dir(Corpus_address + '\\negative')
Train_neg_corpus_address , Test_neg_corpus_address = Partition(Neg_corpus_address)
Train_neg_corpus_content = analysis_dir(Train_neg_corpus_address)
Train_neg_corpus_list = analysis_content(Train_neg_corpus_content)
Neg_dic = Transform(Train_neg_corpus_list)

Test_label = [1]*len(Test_pos_corpus_address) + [0]*len(Test_neg_corpus_address)
Test_corpus_address = Test_pos_corpus_address + Test_neg_corpus_address
Test_data = [(x, y) for x, y in zip(Test_corpus_address, Test_label)]
score = 0
Interference_term = []

for data in Test_data:

    Pos_probability = 0
    Neg_probability = 0
    data_address = data[0]
    data_label = data[1]

    data_content = analysis_file(data_address)
    data_content = re.sub(r'https?:\/\/.*[\r\n]*', '', data_content)
    data_content = re.sub(r'[0-9]', '', data_content)
    word_list = jieba.lcut(sentence = data_content)
    word_clearn_list = []

    for word in word_list:
        if(word not in Stopword and word not in string.punctuation and word != ' '):
            word_clearn_list.append(word)

    for word in word_clearn_list:

        if word in Pos_dic.keys():
            Pos_probability += Pos_dic[word]

        if word in Neg_dic.keys():
            Neg_probability += Neg_dic[word]
    
    difference = Pos_probability - Neg_probability

    if (abs(difference) < 0.01):
        Interference_term.append((word_list , data_label))
        continue

    if (difference >= 0):
        if(data_label == 1):
            score += 1

    else:
        if(data_label == 0):
            score += 1

    # print('% + %',difference, data_label)

# write_txt(Interference_term, 'Interference')
score = score/(len(Test_data) - len(Interference_term))
print('基于朴素贝叶斯模型的情感分析系统准确率为',score)