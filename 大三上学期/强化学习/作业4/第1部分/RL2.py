import numpy as np
import MDP
from sympy import *
import math

class RL2:
    def __init__(self, mdp, sampleReward):

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        # reward 可能为负值
        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def sampleSoftmaxPolicy(self, policyParams, state):
        nActions = self.mdp.nActions
        exp = list(np.exp(policyParams[:, state]))
        action_prob = [i / sum(exp) for i in exp]
        action = np.random.choice(nActions , p = action_prob)
        return action


    def epsilonGreedyBandit(self, nIterations):
        # empiricalMeans 每个动作的经验收益平均值
        R = self.mdp.R
        reward_list = []
        nActions = self.mdp.nActions
        action_count = np.zeros(nActions)
        empiricalMeans = np.zeros(nActions)

        for iteration in range(1 , nIterations + 1):
            epsilon = 1 / iteration
            reward = 0
            if np.random.random() <= epsilon:
                # np.random.randint 左闭右开
                action = np.random.randint(nActions)
            else:
                action = np.argmax(empiricalMeans)

            action_count[action] += 1
            reward = self.sampleReward(R[action])
            reward_list.append(reward)

            # 更新每个动作的经验收益平均值
            empiricalMeans[action] = (empiricalMeans[action] * (action_count[action] - 1) + reward) / action_count[action]

        return empiricalMeans, reward_list

        
    def thompsonSamplingBandit(self, prior, nIterations, k = 1):

        R = self.mdp.R
        reward_list = []
        nActions = self.mdp.nActions
        
        for _ in range(1 , nIterations + 1):
            empiricalMeans = np.zeros(nActions)
            for action_ergodic in range(0, nActions):
                sample_reward = 0
                # 默认k = 1 采样次数为 1
                for _ in range(k):
                    sample_reward += np.random.beta(prior[action_ergodic, 0], prior[action_ergodic, 1])
                empiricalMeans[action_ergodic] = sample_reward / k
            
            action = np.argmax(empiricalMeans)
            reward = self.sampleReward(R[action])
            reward_list.append(reward)

            # 更新参数
            prior[action, 0] += reward
            prior[action, 1] += 1 - reward 
            
        empiricalMeans = np.zeros(nActions)
        for action_ergodic in range(0, nActions):
            sample_reward = 0
            for _ in range(k):
                sample_reward += np.random.beta(prior[action_ergodic, 0], prior[action_ergodic, 1])
            empiricalMeans[action_ergodic] = sample_reward / k
        
        return empiricalMeans, reward_list

    def UCBbandit(self, nIterations):
        R = self.mdp.R
        reward_list = []
        nActions = self.mdp.nActions
        action_count = np.zeros(nActions)
        empiricalMeans = np.zeros(nActions)
        UB = np.zeros(nActions)

        for iteration in range(1 , nIterations + 1):
            reward = 0
            for action_ergodic in range(0, nActions):
                UB[action_ergodic] = empiricalMeans[action_ergodic] + np.sqrt((2 * np.log(iteration)) / action_count[action_ergodic])
        
            action = np.argmax(UB)
            action_count[action] += 1
            reward = self.sampleReward(R[action])
            reward_list.append(reward)

            # 更新每个动作的经验奖励平均值
            empiricalMeans[action] = (empiricalMeans[action] * (action_count[action] - 1) + reward) / action_count[action]

        return empiricalMeans, reward_list
           

    def reinforce(self, s0, initialPolicyParams, nEpisodes, nSteps):
        gamma = 0.95
        alpha = 0.01
        policyParams = initialPolicyParams
        rewardList = []
        nActions = self.mdp.nActions

        for _ in range(0, nEpisodes):
            state = s0
            data = []
            total_reward = 0
            G = np.zeros(nSteps)

            for step in range(0, nSteps):
                action = self.sampleSoftmaxPolicy(policyParams, state)
                reward, next_state = self.sampleRewardAndNextState(state, action)
                total_reward += reward
                data.append([state, action, reward])
                state = next_state
            rewardList.append(total_reward)

            for step in range(0, nSteps):
                G[step] = sum([gamma ** t * data[step + t][2] for t in range(0, nSteps - step)])

                for action in range(0, nActions):
                    if action == data[step][1]:
                        policyParams[action , data[step][0]] += alpha * (gamma ** step) * G[step] * (1 - np.exp(policyParams[action , data[step][0]]) / sum(np.exp(policyParams[:, data[step][0]])))
                    else:
                        policyParams[action , data[step][0]] -= alpha * (gamma ** step) * G[step] * np.exp(policyParams[action , data[step][0]]) / sum(np.exp(policyParams[:, data[step][0]]))
                     
        return [policyParams, rewardList]