import csv
import jieba
import time
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm



INTERVAL = '\t' 
END = '\n' 
sequence_length = 60
lr = 1e-4
epochs = 5
batch_size = 64


class TextGengerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nlayers=1, bidirectional=True, dropout=0.2):
        super(TextGengerator, self).__init__()
        self.directions = 2 if bidirectional else 1
        # 词嵌入层
        self.embed = nn.Embedding(input_size , hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, nlayers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * self.directions, output_size)

    def forward(self, x):
        x = self.embed(x)
        # x 输出 hidden 隐藏状态
        x, hidden = self.gru(x)
        x = self.dropout(x)
        x = self.linear(x)
    
        return x

def load_train_dataset(file_path):
    train_dataset = ''
    with open(file_path) as file:
        csv_reader = csv.reader(file)
        for i , row in enumerate(csv_reader):
            if i == 0:
                continue
            train_dataset = train_dataset + row[2] + INTERVAL \
                                          + row[3] + INTERVAL \
                                          + row[4] + INTERVAL \
                                          + row[5] + INTERVAL \
                                          + row[6] + END
    return train_dataset

def create_dictionary(train_dataset):
    vocab = sorted(set(train_dataset))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    return char2idx, idx2char

def conversion_train_dataset(train_dataset):
    global char2idx, idx2char
    train_dataset = train_dataset.split(END)[:-1]
    train_dataset_conversion = []

    for story in (train_dataset):
        num = len(story) // sequence_length
        for i in range(num):
            data = story[i * sequence_length : i * sequence_length + sequence_length]
            input_data = data[:-1]
            target_data = data[1:]
            input_sequence = np.array([char2idx[char] for char in input_data])
            target_sequence = np.array([char2idx[char] for char in target_data])
            train_dataset_conversion.append((input_sequence, target_sequence))

    return train_dataset_conversion

def load_test_dataset(file_path):
    test_dataset = []
    with open(file_path) as file:
        csv_reader = csv.reader(file)
        for i , row in enumerate(csv_reader):
            if i == 0:
                continue
            test_dataset.append(row[2])
    return test_dataset

def conversion_test_dataset(test_dataset):
    global char2idx, idx2char
    test_dataset_conversion = []

    for data in test_dataset:
        sequence = np.array([char2idx[char] for char in data])
        test_dataset_conversion.append(sequence)

    return test_dataset_conversion

def train(model, train_data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):

        losses = 0

        model.train()
        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

        train_bar = tqdm(train_dataloader , ncols = 100)
        for sequence, target in train_bar:

            model.zero_grad()
            output = model(sequence.to(device))

            loss = criterion(output.permute(0, 2, 1), target.long().to(device))
            losses += loss.item()

            loss.backward()
            optimizer.step()

        average_loss = losses / len(train_dataloader)
        print('\tTrain Loss:', average_loss)
    
    torch.save(model.state_dict(), 'GRU_params.pth')

def genarator(model, test_sentence_dataset, vocab_size):
    global char2idx, idx2char

    temperature = 100.
    story_generator = ''
    sequence = test_dataset_conversion[0]

    # 加载预训练模型
    model.load_state_dict(torch.load("GRU_params.pth"))
    model.eval()
    with torch.no_grad():
        count = 0
        while True:
            output = model(torch.tensor(sequence).unsqueeze(0).to(device))
            pred = output.squeeze(0)[-1,:].cpu().numpy() / temperature

            # pred_idx = np.argmax(pred)
            pred -= np.max(pred)
            pred = np.exp(pred) / np.sum(np.exp(pred))
            pred_idx = np.random.choice(vocab_size, p=pred)

            if idx2char[pred_idx] == INTERVAL:
                count += 1
            if count == 6:
                break

            story_generator = story_generator + idx2char[pred_idx]
            # nn.GRU nn.LSTM 没有输入隐藏状态时, 初始化为零
            sequence = np.append(sequence, [pred_idx])
    
    print(story_generator)


# 数据预处理
train_dataset_path = "./story_genaration_dataset/ROCStories_train.csv"
train_dataset = load_train_dataset(train_dataset_path)
char2idx, idx2char = create_dictionary(train_dataset)
train_dataset_conversion = conversion_train_dataset(train_dataset)

vocab_size = len(char2idx)

test_dataset_path = './story_genaration_dataset/ROCStories_test.csv'
test_dataset = load_test_dataset(test_dataset_path)
test_dataset_conversion = conversion_test_dataset(test_dataset)
print("数据加载完毕!")

# 定义模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TextGengerator(input_size = vocab_size, hidden_size = 100, output_size = vocab_size).to(device)

# 训练模型
# train(model, train_dataset_conversion)
# 生成
genarator(model, test_dataset_conversion, vocab_size)