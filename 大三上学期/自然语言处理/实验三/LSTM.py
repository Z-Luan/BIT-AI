import csv
import jieba
import time
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 超参数设置
END = 'zl1120202786'
INTERVAL = '\n'
sequence_length = 15
lr = 1e-4
epochs = 2
batch_size = 128


class TextGengerator(nn.Module):
    def __init__(self, vocab_size, ninp = 500, nhid = 200, nlayers = 1, dropout = 0.2):
        super(TextGengerator, self).__init__()
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size , ninp)
        self.lstm = nn.LSTM(input_size = ninp, hidden_size = nhid, num_layers = nlayers, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        # 构造分类器
        self.lay1 = nn.Sequential(nn.Linear(2 * nhid , 128),
                                  nn.ReLU(),
                                  nn.Dropout(dropout))
        self.lay2 = nn.Linear(128 , vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        # output, (hn, cn) = lstm(inputs)
        # output (batch, seq_len, num_directions * hidden_size)
        # h_n (num_layers * num_directions, batch, hidden_size) 储存隐藏状态信息, 即输出信息
        # c_n (num_layers * num_directions, batch, hidden_size) 储存单元状态信息
        x = self.lstm(x)[0]
        x = self.dropout(x)
        x = self.lay1(x)
        x = self.lay2(x)

        return x


def cut_sentence(sentence):
    sentence_cut = jieba.lcut(sentence)
    while ' ' in sentence_cut:
        sentence_cut.remove(' ')
    return sentence_cut

def load_train_dataset(file_path):
    train_dataset = []
       
    with open(file_path) as file:
        csv_reader = csv.reader(file)
        for i , row in enumerate(csv_reader):
            if i == 0:
                continue
            data = cut_sentence(row[2]) + [INTERVAL] + \
                   cut_sentence(row[3]) + [INTERVAL] + \
                   cut_sentence(row[4]) + [INTERVAL] + \
                   cut_sentence(row[5]) + [INTERVAL] + \
                   cut_sentence(row[6]) + [END]

            # 抽样训练
            # if i >= 10000:
            #     break

            train_dataset.append(data)
    return train_dataset


def create_dictionary(train_dataset):
    word2tkn = {}
    tkn2word = []
    for data in train_dataset:
        for word in data:
            if word not in word2tkn:
                tkn2word.append(word)
                word2tkn[word] = len(tkn2word) - 1
    return word2tkn, tkn2word


def conversion_train_dataset(train_dataset):
    global word2tkn, tkn2word
    train_dataset_conversion = []

    for story in (train_dataset):
        for i in range(len(story) - sequence_length - 1):
            try:
               input =  story[i : i + sequence_length]
               input = np.array([word2tkn[word] for word in input])
               target = story[i + 1 : i + sequence_length + 1]
               target = np.array([word2tkn[word] for word in target])

               train_dataset_conversion.append((input, target))
            except:
                pass
        
    return train_dataset_conversion


def train(model, train_data):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):

        # 训练数据集的损失率
        losses = 0

        model.train()
        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

        # 进度条设置
        train_bar = tqdm(train_dataloader , ncols = 100)
        for sequence, target in train_bar:
            # 梯度清零
            model.zero_grad()

            output = model(sequence.to(device))

            # 计算损失
            loss = criterion(output.permute(0, 2, 1), target.long().to(device))
            losses += loss.item()

            # 模型更新
            loss.backward()
            optimizer.step()

        average_loss = losses / len(train_dataloader)
        print('\tTrain Loss:', average_loss)
    
    torch.save(model.state_dict(), 'LSTM1_5_params.pth')


def load_test_dataset(file_path):
    test_dataset = []

    with open(file_path) as file:
        csv_reader = csv.reader(file)
        for i , row in enumerate(csv_reader):
            data = []
            if i == 0:
                continue
            data = cut_sentence(row[2])
            test_dataset.append(data)
    return test_dataset

def conversion_test_dataset(test_dataset):
    global word2tkn, tkn2word
    test_dataset_conversion = []

    for data in test_dataset:
        sequence = []
        for word in data:
            if word in word2tkn:
                sequence.append(word2tkn[word])
            else:
                continue
        test_dataset_conversion.append(np.array(sequence))
    
    return test_dataset_conversion


def genarator(model, test_sentence_dataset, vocab_size):
    temperature = 1.0
    story_generator = ''
    sequence = test_dataset_conversion[0]

    # 加载预训练模型
    model.load_state_dict(torch.load("LSTM1_15_params.pth"))

    model.eval()
    with torch.no_grad():
        count = 0
        while True:
            # if len(sequence) > sequence_length:
            #     sequence = sequence[-5:]
            
            output = model(torch.tensor(sequence).unsqueeze(0).to(device))
            
            # pred_idx = torch.argmax(output.squeeze(0)[-1,:]).item()
            pred = output.squeeze(0)[-1,:].cpu().numpy() / temperature

            # softmax 避免指数溢出
            pred -= np.max(pred)
            pred = np.exp(pred) / np.sum(np.exp(pred))
            pred_idx = np.random.choice(vocab_size, p=pred)

            if tkn2word[pred_idx] == INTERVAL:
                count += 1
            if count == 5:
                break
            
            story_generator = story_generator + ' ' + tkn2word[pred_idx]
            
            # sequence = np.append(sequence[1:], [pred_idx])
            sequence = np.append(sequence, [pred_idx])

    print(story_generator)


# 数据预处理
train_dataset_path = "./story_genaration_dataset/ROCStories_train.csv"
train_dataset = load_train_dataset(train_dataset_path)
word2tkn, tkn2word = create_dictionary(train_dataset)
# 词汇表大小
vocab_size = len(word2tkn)
train_dataset_conversion = conversion_train_dataset(train_dataset)

test_dataset_path = './story_genaration_dataset/ROCStories_test.csv'
test_dataset = load_test_dataset(test_dataset_path)
test_dataset_conversion = conversion_test_dataset(test_dataset)
print("数据加载完毕!")

# 定义模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TextGengerator(vocab_size).to(device)

# 训练模型
# train(model, train_dataset_conversion)
# 生成
genarator(model, test_dataset_conversion, vocab_size)