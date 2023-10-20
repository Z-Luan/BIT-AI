from email import policy
import math
import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        reward = self.sampleReward(self.mdp.R[action,state]) 
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        
        Q = initialQ
        Action_num , State_num = Q.shape
        rewardList = np.zeros(nEpisodes)
        n = np.zeros((State_num , Action_num))
        for episode in range(nEpisodes):
            Total_reward = 0
            state = s0
            for step in range(nSteps):
                if(np.random.uniform() < epsilon):
                    action = np.random.randint(0 , Action_num)
                elif temperature == 0:
                    action = np.argmax(Q[:, state])
                else:
                    Q_state = Q[:, state]
                    P_action = np.exp(Q_state / temperature) / np.sum(np.exp(Q_state / temperature))
                    action = np.random.choice(Action_num , p=P_action)

                reward , nextState = self.sampleRewardAndNextState(state , action)
                n[state , action] += 1
                alpha = 1 / n[state , action]
                Q[action , state] += alpha * (reward + self.mdp.discount * np.max(Q[:, nextState]) - Q[action , state])
                state = nextState
                Total_reward += reward                      

            rewardList[episode] = Total_reward

        policy = np.argmax(Q , axis=0)

        return [Q,policy,rewardList]