import torch
import torch.nn as nn
import time

from torch.utils.data import  DataLoader
from Exp_DataSet import Corpus
from Exp_Model import BiLSTM_model, Transformer_model


def train():
    #  完成一个 epoch 的训练

    sum_true = 0
    sum_loss = 0.0

    max_valid_acc = 0

    model.train()
    for data in data_loader_train:
        # 选取对应批次数据的输入和标签
        batch_x, batch_y = data[0].to(device), data[1].to(device)

        # 模型预测
        y_hat = model(batch_x)

        loss = loss_function(y_hat, batch_y)
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
        sum_true += torch.sum(y_hat == batch_y).float()
        sum_loss += loss.item()

    train_acc = sum_true / dataset.train.__len__()
    train_loss = sum_loss / (dataset.train.__len__() / batch_size)

    valid_acc = valid()

    if valid_acc > max_valid_acc:
        torch.save(model, "checkpoint.pt")

    print(f"epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc*100:.2f}%, valid accuracy: {valid_acc*100:.2f}%,\
            time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) }")
    
    txt = '\tTrain ACC:' +'\t' + str(train_acc * 100) + '\tTrain Loss:' + str(train_loss) +'\tValid ACC:' + str(valid_acc * 100) + '\n'

    # 写入记录文件
    file.write(txt)


def valid():
    # 进行验证，返回模型在验证集上的 accuracy 

    sum_true = 0

    model.eval()
    with torch.no_grad():
        for data in data_loader_valid:
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            y_hat = model(batch_x)

            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)

            sum_true += torch.sum(y_hat == batch_y).float()

        return sum_true / dataset.valid.__len__()


def predict():
    # 读取训练好的模型对测试集进行预测，并生成结果文件
    
    results = []

    model = torch.load('checkpoint.pt').to(device)
    model.eval()
    with torch.no_grad():
        for data in data_loader_test:
            batch_x = data[0].to(device)

            y_hat = model(batch_x)
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat])

            results += y_hat.tolist()

    # 写入预测文件
    with open("predict.txt", "w") as f:
        for label_idx in results:
            label = dataset.dictionary.idx2label[label_idx][1]
            f.write(label+"\n")


if __name__ == '__main__':
    dataset_folder = 'data/tnews_public'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    max_sent_len = 150
    batch_size = 16
    epochs = 6
    lr = 1e-4
    embedding_random = False

    dataset = Corpus(dataset_folder, max_sent_len)
    vocab_size = len(dataset.dictionary.tkn2word)

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    # model = Transformer_model(vocab_size , embedding_weight = dataset.embedding_weight , embedding_random_flag = embedding_random).to(device)
    model = BiLSTM_model(vocab_size , embedding_weight = dataset.embedding_weight , embedding_random_flag = embedding_random).to(device)
    # model.embed.weight.data.copy_(torch.from_numpy(dataset.embedding_weight))
    
    
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()                                      
    # 设置优化器 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) 

    # 进行训练
    file = open('BiLSTM_record_file.txt' , 'w', encoding='utf-8')
    for epoch in range(epochs):
        train()
    file.close()
    # 对测试集进行预测
    predict()
