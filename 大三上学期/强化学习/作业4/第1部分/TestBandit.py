import numpy as np
import MDP
import RL2


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli

    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean:
        return 1
    else:
        return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]], [[1]], [[1]]])
R = np.array([[0.3], [0.5], [0.7]])
discount = 0.999
mdp = MDP.MDP(T, R, discount)
banditProblem = RL2.RL2(mdp, sampleBernoulli)

# Test epsilon greedy strategy
rewardList_sum_epsilon = np.zeros(200)
for i in range(1000):
    empiricalMeans,reward_list_epsilon = banditProblem.epsilonGreedyBandit(nIterations=200)
    rewardList_sum_epsilon += reward_list_epsilon
print("\nepsilonGreedyBandit results")
print("epsilonGreedyBandit results",rewardList_sum_epsilon/1000)


# Test Thompson sampling strategy
# 先验设置为1
rewardList_sum_thompson = np.zeros(200)
for i in range(1000):
    empiricalMeans,reward_list_thompson = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions, 2]), nIterations=200)
    rewardList_sum_thompson += reward_list_thompson
print("\nthompsonSamplingBandit results")
print("epsilonGreedyBandit results",rewardList_sum_thompson/1000)

# Test UCB strategy
rewardList_sum_UCB = np.zeros(200)
for i in range(1000):
    empiricalMeans,reward_list_UCB = banditProblem.UCBbandit(nIterations=200)
    rewardList_sum_UCB += reward_list_UCB
print("\nUCBbandit results")
print("epsilonGreedyBandit results",rewardList_sum_UCB/1000)

import matplotlib.pyplot as plt
fig = plt.figure()
X = np.arange(1,201)
plt.plot(X,rewardList_sum_epsilon/1000,label="epsilon")
plt.plot(X,rewardList_sum_thompson/1000,label="thompson",linestyle="--")
plt.plot(X,rewardList_sum_UCB/1000,label="UCB",linestyle="-.")
plt.legend()
plt.savefig('Bandits.jpg')
plt.show()