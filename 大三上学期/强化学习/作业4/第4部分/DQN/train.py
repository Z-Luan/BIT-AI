import os
import sys
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
from memory import Memory
from env import partial_env_class
from config import initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr, sequence_length, max_epsilon
from collections import deque
import matplotlib
import matplotlib.pyplot as plt




def get_action(state_series, online_net, epsilon, partial_env):
    # This function gives an action for a corresponding state based on the epsilon-greedy exploration strategy. 
    # state_series represents the state variable which contains the last 4 observations. 
    # Epsilon is the exploration variable. 
    # Partial_env represents the environment variable.
    # Online net is the Q network from which the actions need be obtained. 


    if np.random.rand() <= epsilon or len(state_series) < sequence_length:
        return partial_env.get_random_action()
    else:
        return online_net.get_action(torch.stack(list(state_series)))

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main():
    # Perform training and plot results. 
    # Since this is a partially observable environment, the state variable actually represents the observation. 

    partial_env = partial_env_class()
    partial_env.set_seed(500)
    torch.manual_seed(500)

    num_inputs = partial_env.num_states()
    num_actions = partial_env.num_actions()
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    score_list = []
    for e in range(10000):
        done = False

        state_series = deque(maxlen=sequence_length)
        next_state_series = deque(maxlen=sequence_length)
        score = 0
        state = partial_env.reset() 
        state = torch.Tensor(state).to(device)
        running_step = 0
        while not done and running_step<500:
            steps += 1
            state_series.append(state)
            action = get_action(state_series, online_net, epsilon, partial_env)
            next_state, reward, done = partial_env.step(action)

            next_state = torch.Tensor(next_state).to(device)

            next_state_series.append(next_state)
            mask = 0 if done else 1
            
            action_one_hot = np.zeros(2)
            action_one_hot[action] = 1
            if len(state_series) >= sequence_length:
                memory.push(state_series, next_state_series, action_one_hot, reward, mask)

            score += reward
            state = next_state
            running_step += 1

            if steps > initial_exploration:
                epsilon -= 0.000005
                epsilon = max(epsilon, max_epsilon)

                batch = memory.sample(batch_size)
                QNet.train_model(online_net, target_net, optimizer, batch)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        score_list.append(score)
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, score, epsilon))


    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('reward')
    moving_averages= []
    i = 0
    while i < len(score_list) - 100 + 1:
        this_window = score_list[i : i + 100]
        window_average = sum(this_window) / 100
        moving_averages.append(window_average)
        i += 1
    Ep_arr = np.array(moving_averages)
    plt.plot(Ep_arr)
    plt.savefig('./dqncartpole.png')



if __name__=="__main__":
    main()
