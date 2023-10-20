import os
import random
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from game.flappy_bird import GameState
import csv

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 两个动作
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        # 经验回放池大小
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        # PyTorch 与 Tensorflow 不同, 不需要手动定义权重参数和偏置参数
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        # nn.ReLU(inplace=True) inplace为True, 将会改变原始输入数据, 否则不会改变原始输入数据, 只会产生新的输出
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        # nn.Linear(in_features, out_features, bias=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        # -1代表动态调整这个维度上的元素个数, 保证总元素个数不变
        # (batchsize，channels，x，y)
        # out.size()[0]指batchsize的值
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        # torch.nn.init.uniform(torch.Tensor, a, b) 从均匀分布中采样填充输入向量
        # a 均匀分布的下界
        # b 均匀分布的上界
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    # np.transpose(2,0,1) 把 array 的第2个参数变为第0个参数, 第0个参数变为第1个参数, 第1个参数变为第2个参数
    image_tensor = image.transpose(2, 0, 1)
    # np.astype 转化 array 的数据类型
    image_tensor = image_tensor.astype(np.float32)
    # torch.from_numpy 将 array 转化为张量
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available(): # 是否支持 CUDA
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    # cv2.cvtColor(iamge, cv2.COLOR_BGR2GRAY) 转化为黑白二值图像
    # cv2.resize 修改原始图像尺寸
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    # np.reshape 规格转换, 需要保证元素总个数一致
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def train(train_network, target_network, start):

    rewardList = []
    Total_reward = 0
    episode = 0

    # 定义 Adam 优化器
    # train_network.parameters() 保存了 Weights 和 Bais 参数值
    optimizer = optim.Adam(train_network.parameters(), lr=1e-6)

    # 初始化均方损失误差
    criterion = nn.MSELoss()

    # 初始化 flappy bird 游戏
    game_state = GameState()

    # 初始化经验回放池
    replay_memory = []

    # initial action is do nothing
    # torch.float32 tensor 数据类型
    # torch.zeros(size, dtype) 返回一个形状为 size, 数据类型为 torch.dtype 的tensor
    action = torch.zeros([train_network.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    # torch.cat 在给定维度上对输入的张量序列进行连接操作
    # .unsqueeze(0) 在第 0 个位置增加维度
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # 初始化 epsilon 
    epsilon = train_network.initial_epsilon
    iteration = 0

    # numpy.linspace()在线性空间中以均匀步长生成数字序列
    epsilon_decrements = np.linspace(train_network.initial_epsilon, train_network.final_epsilon, train_network.number_of_iterations)

    while iteration < train_network.number_of_iterations:

        if(iteration % 100 == 0):
            target_network.load_state_dict(train_network.state_dict())

        # 从神经网络获得输出
        output = train_network(state)[0]

        # 初始化动作
        action = torch.zeros([train_network.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("执行随机动作!")

        # torch.randint 返回填充了采样数的张量
        # 采样数的取值范围是[0, train_network.number_of_actions)
        # 采样数的数据类型是 torch.int
        # torch.Size([]) 定义输出张量形状
        action_index = [torch.randint(train_network.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        action[action_index] = 1

        # 获得下一状态和及时奖励
        image_data_1, reward, terminal = game_state.frame_step(action)

        if terminal:
            episode += 1
            rewardList.append((episode, Total_reward))
            Total_reward = 0
        elif reward == 1:
            Total_reward += reward
        else:
            reward += 0

        # print(reward, terminal)
        # print(rewardList)
        # time.sleep(0.1)
        
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        # torch.from_numpy 将数组转化为张量
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # 放回经验回放池
        replay_memory.append((state, action, reward, state_1, terminal))

        # 如果经验回放池已满, 替换在回放池中时间最长的数据
        if len(replay_memory) > train_network.replay_memory_size:
            replay_memory.pop(0)

        # epsilon 退火
        epsilon = epsilon_decrements[iteration]

        # 随机采样 minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), train_network.minibatch_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # 获得下一状态神经网络的输出
        output_1_batch = target_network(state_1_batch)

        # y_j to r_j for terminal state, otherwise to r_j + gamma * max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + train_network.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # 提取 Q-value
        q_value = torch.sum(train_network(state_batch) * action_batch, dim = 1)

        optimizer.zero_grad()

        # 返回一个新的张量，从当前图中分离出来，结果不需要计算梯度
        y_batch = y_batch.detach()

        # 计算损失误差
        loss = criterion(q_value, y_batch)

        # 梯度下降反向传播
        loss.backward()
        optimizer.step()

        state = state_1
        iteration += 1

        if iteration % 25000 == 0:
            torch.save(train_network, "target_dqn_pretrained_train_network_2/current_train_network_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))
    
    with open('target_dqn_reward_2.csv', 'w') as csvfile:
        writer  = csv.writer(csvfile)
        for data in rewardList:
            writer.writerow(data)
    
def test(model):
    game_state = GameState()

    # 初始化动作
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # 获得神经网络输出
        output = model(state)[0]
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        if torch.cuda.is_available(): 
            action = action.cuda()

        action_index = torch.argmax(output)
        if torch.cuda.is_available():  
            action_index = action_index.cuda()
        action[action_index] = 1

        # 执行下一动作
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        state = state_1


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'target_dqn_pretrained_train_network_2/current_train_network_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available: 
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('target_dqn_pretrained_train_network_2/'):
            os.mkdir('target_dqn_pretrained_train_network_2/')

        train_network = NeuralNetwork()
        target_network = NeuralNetwork()

        if cuda_is_available: 
            train_network = train_network.cuda()
            target_network = target_network.cuda()

        train_network.apply(init_weights)
        start = time.time()

        train(train_network, target_network, start)


if __name__ == "__main__":
    main(sys.argv[1])
