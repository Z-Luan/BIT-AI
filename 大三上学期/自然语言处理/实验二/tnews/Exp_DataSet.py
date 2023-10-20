import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
import time
import jieba

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

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1


class Corpus(object):
    def __init__(self, path, max_sent_len):
        self.dictionary = Dictionary(path)
        
        self.max_sent_len = max_sent_len

        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)

        self.embedding_weight = None

        Embedding_File_Path = "sgns.baidubaike.bigram-char"

        Dic = {}
        with open(Embedding_File_Path, 'r', encoding = 'utf-8') as f:
            for i, line in enumerate (f.readlines()):
                line = line.strip().replace(u'\u3000',u'')
                if i == 0:
                    continue
                word_msg = line.split()
                word_name = word_msg[0]
                word_vec = word_msg[1:]
                word_vec = np.array(list(map(float, word_vec)))
                Dic[word_name] = word_vec
        
        len_word_vec = 300

        # 构造映射矩阵
        Mapping_matrix = np.zeros((len(self.dictionary.tkn2word), len_word_vec) , dtype = 'float32')
        for tkn , word in enumerate(self.dictionary.tkn2word):
            vec = Dic.get(word , np.zeros(300))
            Mapping_matrix[tkn] = vec
        
        self.embedding_weight =  torch.tensor(np.array(Mapping_matrix))


    def pad(self, origin_sent):

        if len(origin_sent) > self.max_sent_len:
            return origin_sent[:self.max_sent_len]
        else:
            return origin_sent + [0 for _ in range(self.max_sent_len-len(origin_sent))]


    def tokenize(self, path, test_mode = False):
        
        with open(path, 'r', encoding = 'utf8') as f:
            if test_mode:
                idss = []
                for line in f:
                    one_data = json.loads(line)
                    sent = one_data['sentence']
                    sent = jieba.lcut(sent)

                    for word in sent:
                        self.dictionary.add_word(word)

                    ids = []
                    for word in sent:
                        ids.append(self.dictionary.word2tkn[word])
                    idss.append(self.pad(ids))

                idss = torch.tensor(np.array(idss))

                # TensorDataset 对 tensor 进行打包
                return TensorDataset(idss)

            else:
                idss = []
                labels = []
                for line in f:
                    # 读取一条数据
                    one_data = json.loads(line)
                    sent = one_data['sentence']
                    label = one_data['label']
                    sent = jieba.lcut(sent)
                    # 向词典中添加词
                    for word in sent:
                        self.dictionary.add_word(word)

                    ids = []
                    for word in sent:
                        ids.append(self.dictionary.word2tkn[word])
                    # padding: 将一个 sentence 补 0 至预设的最大句长 self.max_sent_len, 对于句长大于 self.max_sent_len 的句子进行截取
                    idss.append(self.pad(ids))
                    labels.append(self.dictionary.label2idx[label])

                idss = torch.tensor(np.array(idss))
                labels = torch.tensor(np.array(labels)).long()

                # TensorDataset 对 tensor 进行打包
                return TensorDataset(idss, labels)