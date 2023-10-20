import re
import string
from turtle import pd
import numpy as np
import os
import jieba
import pandas as pd
import math
import random


def Partition(file_list_address):
    Train_file_list_address = random.sample(file_list_address, math.ceil(len(file_list_address)*0.8))
    Test_file_list_address =  [i for i in file_list_address if i not in Train_file_list_address]
    return Train_file_list_address , Test_file_list_address

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
    

def analysis_file(file_address):
    content = ''
    if (os.path.exists(file_address)):
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


def Radio(length):
    return (1/length + 1)


Stopword = [line.strip() for line in open('C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data\\vocabulary\\stopwords_utf8.txt', 'r',encoding='utf-8').readlines()]
Pos_list = [line.strip() for line in open('C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data\\vocabulary\\full_pos_dict_sougou.txt', 'r',encoding='utf-8').readlines()]
Neg_list = [line.strip() for line in open('C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data\\vocabulary\\full_neg_dict_sougou.txt', 'r',encoding='utf-8').readlines()] 
Degree_pd = pd.read_excel('C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data\\vocabulary\\degree_dict.xlsx')


Corpus_address = "C:\\Users\\16176\\Desktop\\NLP\\NLP 实践1 参考资源\\data"

Pos_corpus_address = get_dir(Corpus_address + '\\positive')
Train_pos_corpus_address , Test_pos_corpus_address = Partition(Pos_corpus_address)
Train_pos_corpus_content = analysis_dir(Train_pos_corpus_address)
Train_pos_corpus_list = analysis_content(Train_pos_corpus_content)

Neg_corpus_address = get_dir(Corpus_address + '\\negative')
Train_neg_corpus_address , Test_neg_corpus_address = Partition(Neg_corpus_address)
Train_neg_corpus_content = analysis_dir(Train_neg_corpus_address)
Train_neg_corpus_list = analysis_content(Train_neg_corpus_content)


Test_label = [1]*len(Test_pos_corpus_address) + [0]*len(Test_neg_corpus_address)
Test_corpus_address = Test_pos_corpus_address + Test_neg_corpus_address
Test_data = [(x, y) for x, y in zip(Test_corpus_address, Test_label)]
score = 0
Interference_num = 0

for data in Test_data:

    Pos_score = 0
    Neg_score = 0
    data_corpus_address = data[0]
    data_label = data[1]

    data_corpus_content = analysis_file(data_corpus_address)
    data_corpus_content = re.sub(r'https?:\/\/.*[\r\n]*', '', data_corpus_content)
    data_corpus_content = re.sub(r'[0-9]', '', data_corpus_content)
    data_corpus_list = jieba.lcut(sentence = data_corpus_content)
    data_clearn_corpus_list = []
    
    for word in data_corpus_list:
        if(word not in Stopword and word not in string.punctuation and word != ' '):
            data_clearn_corpus_list.append(word)
    
    Adv_pos_visit = [1] * len(data_clearn_corpus_list)
    Adv_neg_visit = [1] * len(data_clearn_corpus_list)
    
    for word in data_clearn_corpus_list:
    
        if word in Pos_list:
            Pos_score += 1
            word_index = data_clearn_corpus_list.index(word)
            for i in range(word_index):
                if (data_clearn_corpus_list[i] in Degree_pd['word'].values.tolist() and Adv_pos_visit[i] == 1):
                    Adv_pos_visit[i] = 0
                    # print(Adv_pos_visit)
                    lengh = word_index - i
                    radio = Radio(lengh)
                    Pos_score = radio * Pos_score * Degree_pd.strength[Degree_pd[Degree_pd.word == data_clearn_corpus_list[i]].index.tolist()[0]]

        if word in Neg_list:
            Neg_score += 1
            word_index = data_clearn_corpus_list.index(word)
            for i in range(word_index):
                if (data_clearn_corpus_list[i] in Degree_pd['word'].values.tolist() and Adv_neg_visit[i] == 1):
                    Adv_neg_visit[i] = 0
                    lengh = word_index - i
                    radio = Radio(lengh)
                    Neg_score = radio * Neg_score * Degree_pd.strength[Degree_pd[Degree_pd.word == data_clearn_corpus_list[i]].index.tolist()[0]]

    difference = Pos_score - Neg_score

    # print('% + %',difference, Pos_score, Neg_score)

    if (difference > 0):
        if(data_label == 1):
            score += 1

    if (difference < 0):
        if(data_label == 0):
            score += 1

    if (difference == 0):
        Interference_num += 1

    # print('% + %',score, data_label)
        
score = score/(len(Test_data) - Interference_num)
print('基于词典和规则的情感分析系统准确率为',score)


                
            
    
    
