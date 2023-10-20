import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import numpy as np
import time
import csv
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable

def select_top_k(pred, k = 10):
    pred_value = F.softmax(pred.sort(descending = True)[0][:k]).cpu().numpy()
    pred_index = pred.sort(descending = True)[1][:k].cpu().numpy()

    return np.random.choice(pred_index , p=pred_value)

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2")


model = GPT2LMHeadModel.from_pretrained("./gpt2")


from torch.utils.data import DataLoader, TensorDataset

# 超参数
sequence_length = 5
batch_size = 128
epochs = 5
lr = 1e-4

def load_train_dataset(file_path):
    train_dataset_vector_list = []
       
    with open(file_path) as file:
        csv_reader = csv.reader(file)
        for i , row in enumerate(csv_reader):
            train_dataset_vector = ''
            if i == 0:
                continue
            train_dataset_vector = row[2] + '\n' + row[3] + '\n' + row[4] + '\n' + row[5] + '\n' + row[6] + '<|endoftext|>'
            train_dataset_vector_list.append(train_dataset_vector)
            # 抽样训练
            # if i >= 1:
            #     break
    return train_dataset_vector_list

def conversion_train_dataset(train_dataset_vector_list):
    train_dataset = []
    for train_dataset_vector in train_dataset_vector_list:
        train_dataset_sequence = tokenizer.encode(train_dataset_vector)
        train_dataset.append(train_dataset_sequence)
    return train_dataset

train_dataset_path = "./story_genaration_dataset/ROCStories_train.csv"
train_dataset_vector_list = load_train_dataset(train_dataset_path)
train_dataset = conversion_train_dataset(train_dataset_vector_list)
print('加载数据完毕!')

# 配置训练方式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.train()
# 不需要计算损失函数, GPT2 模型内嵌
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

for epoch in range(epochs):
    losses = 0 

    start = time.perf_counter()
    for i, data in enumerate(train_dataset):
        # 显示训练进度
        finsh = "▓" * int(i / len(train_dataset) * 100)
        need_do = "-" * (100 - int(i / len(train_dataset) * 100))
        progress = int(i / len(train_dataset) * 100)
        dur = time.perf_counter() - start
        print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(progress, finsh, need_do, dur), end="")

        # 增加 batch 维度
        data = torch.tensor(data).unsqueeze(0)
        # 标签与样本数据相同
        target = data

        # tensor 不能反向传播，variable 可以反向传播
        data, target = Variable(data).to(device), Variable(target).to(device)

        # 梯度清零
        optimizer.zero_grad()
        output = model(data, labels = target)
        loss = output[0]

        losses += loss.item()

        # 模型更新
        loss.backward()
        optimizer.step()

    print('Train loss:', losses / len(train_dataset))

# 保存模型参数
torch.save(model.state_dict(), 'GPT2_2_params.pth')
# 参数载入
# model.load_state_dict(torch.load('GPT2_params.pth'))
# 保存模型
torch.save(model, 'GPT2_2_model.pth')
# 模型载入
# model = torch.load('GPT2_model.pth')


######### 调用 GPT2 生成 #########
# def load_test_dataset(file_path):
#     test_dataset1 = []
#     test_dataset2 = []

#     with open(file_path) as file:
#         csv_reader = csv.reader(file)
#         for i , row in enumerate(csv_reader):
#             if i == 0:
#                 continue
#             test_dataset1.append(row[2])
#             pre_content = row[3] + '\n' + row[4] + '\n' + row[5] + '\n' + row[6] + '<|endoftext|>'
#             test_dataset2.append(pre_content)
#     return test_dataset1, test_dataset2


# test_dataset_path = './story_genaration_dataset/ROCStories_test.csv'
# test_dataset1, test_dataset2 = load_test_dataset(test_dataset_path )

# sequence = tokenizer.encode(test_dataset1[0])
# # 增加 batch 维度
# sequence_tensor = torch.tensor(sequence).unsqueeze(0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # model = torch.load('GPT2_model.pth')
# model.to(device)
# model.eval()
# temperature = 1.0
# for _ in range(50):
#     sequence_tensor = sequence_tensor.to(device)

#     with torch.no_grad():
#         outputs = model(sequence_tensor)
#         pred = outputs[0]

#     pred = pred[0, -1, :] / temperature
#     pred_index = select_top_k(pred, k = 10)
#     pred_text = tokenizer.decode(sequence + [pred_index])

#     if '<|endoftext|>' in pred_text:
#         # 出现文本结束标志，结束文本生成
#         print('End')
#         break

#     sequence += [pred_index]

#     if len(sequence) > 1023:
#         # GPT2 模型最长输入长度为1024，如果长度过长则截断
#         sequence = sequence[-1023:]

#     sequence_tensor = torch.tensor(sequence).unsqueeze(0)

# print(pred_text)