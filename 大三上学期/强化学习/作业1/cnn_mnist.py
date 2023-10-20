import torch
import torch.nn as nn
import pickle
import random
import time
import torch.nn.functional as F


def load_data():
    x_train, y_train, x_test, y_test = pickle.load(open("mnist.pkl", 'rb'))
    x_train, y_train = x_train[:6000], y_train[:6000]
    x_test, y_test = x_test[:1000], y_test[:1000]

    x_train = torch.FloatTensor(x_train) # 32-bit floating point
    y_train = torch.LongTensor(y_train) # 64-bit integer
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test) 
    
    x_train = x_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)
    return x_train, y_train, x_test, y_test


class Cnn_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,5,stride=(1,1),padding=2),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,5,stride=(1,1),padding=2),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.dropout = nn.Dropout2d()#默认p为0.5

        self.bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(7*7*64,1000)
        self.fc2 = nn.Linear(1000,10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.dropout(x)
        x = self.bn(x)
        x = x.view(-1,7*7*64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def train(x_train, y_train, x_test, y_test, BATCH_SIZE, model, loss_function, optimizer):

    train_Num = x_train.shape[0]
    for epoch in range(20):
        batchindex = list(range(int(train_Num / BATCH_SIZE)))
        random.shuffle(batchindex)
        for i in batchindex:
            batch_x = x_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            batch_y = y_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]

            y_pre = model(batch_x)
            loss = loss_function(y_pre, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_pre = model(x_test)

        y_pre = torch.max(y_pre, 1)[1].data.squeeze()
        score = torch.sum(y_pre == y_test).float() / y_test.shape[0]
        print(f"epoch:{epoch},train loss: {loss:.4f}, test accuracy: {score:.4f}, time:{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) }")
    print("ZL's procedure has been completed")


def cnn():
    x_train, y_train, x_test, y_test = load_data()
    BATCH_SIZE = 100
    model = Cnn_net()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(x_train, y_train, x_test, y_test,
          BATCH_SIZE, model, loss_function, optimizer)


cnn()
