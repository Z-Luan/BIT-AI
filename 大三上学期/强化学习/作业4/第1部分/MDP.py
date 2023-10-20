
from email import policy
from re import A
import numpy as np

class MDP:

    def __init__(self,T,R,discount):
        assert T.ndim == 3, "转移函数无效，应该有3个维度"
        self.nActions = T.shape[0] 
        self.nStates = T.shape[1] 
        assert T.shape == (self.nActions,self.nStates,self.nStates), "无效的转换函数：它具有维度 " + repr(T.shape) + ", 但它应该是(nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "无效的转移函数：某些转移概率不等于1"
        self.T = T
        assert R.ndim == 2, "奖励功能无效：应该有2个维度"
        assert R.shape == (self.nActions,self.nStates), "奖励函数无效：它具有维度 " + repr(R.shape) + ", 但它应该是 (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "折扣系数无效：它应该在[0,1]中"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):

        iterId = 0
        epsilon = 0
        V = initialV

        while iterId <= nIterations:
            # R |A| x |S|
            # T |A| x |S| x |S'|
            # V |S'|

            iterId += 1

            V_iteration = np.max(self.R + self.discount * (self.T.dot(V)), axis=0)

            epsilon = np.max(np.fabs(V_iteration - V))
            if(epsilon <= tolerance):
                V = V_iteration
                return [V, iterId, epsilon]

            V = V_iteration

        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        policy = np.argmax(self.R + self.discount * (self.T.dot(V)), axis=0)

        return policy 

    def evaluatePolicy(self,policy):

        T_policy = np.zeros((self.nStates,self.nStates))
        R_policy = np.zeros(self.nStates)

        for i in range (self.nStates):
            T_policy[i] = self.T[policy[i]][i]
            R_policy[i] = self.R[policy[i]][i]

        I = np.eye(self.T.shape[1]) # 生成单位矩阵 
        V = np.dot(np.linalg.inv(I - self.discount * T_policy), R_policy)

        # # 迭代求解
        # tolerance = 0.01
        # # 初始化 V , 经证明任意初始化 V 均可以收敛到 V^pi
        # V = np.zeros(self.nStates)
        # while True:
        #     V_iteration = R_policy + self.discount * (T_policy.dot(V))
        #     epsilon = np.max(np.fabs(V_iteration - V))

        #     if(epsilon <= tolerance):
        #         V = V_iteration
        #         return V
        #     V = V_iteration

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):

        V = np.zeros(self.nStates)
        policy = initialPolicy
        iterId = 0

        while iterId <= nIterations:
            # R |A| x |S|
            # T |A| x |S| x |S'|
            # V |S'|

            V_iteration = self.evaluatePolicy(policy)
            policy_iteration = self.extractPolicy(V_iteration)

            if((policy_iteration == policy).all()):
                policy = policy_iteration
                V = V_iteration
                return [policy, V, iterId]
            
            policy = policy_iteration
            V = V_iteration 
            iterId += 1

        return [policy, V, iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):

        V = initialV
        iterId = 0
        epsilon = 0

        #填空部分
        T_policy = np.zeros((self.nStates,self.nStates))
        R_policy = np.zeros(self.nStates)

        for i in range (self.T.shape[1]):
            T_policy[i] = self.T[policy[i]][i]
            R_policy[i] = self.R[policy[i]][i]
        
        while iterId <= nIterations:

            V_iteration = R_policy + self.discount * (T_policy.dot(V))

            epsilon = np.max(np.fabs(V_iteration-V))
            if(epsilon <= tolerance):
                V = V_iteration
                return [V, iterId, epsilon]

            V = V_iteration
            iterId += 1

        return [V, iterId, epsilon]


    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=0,nIterations=np.inf,tolerance=0.01):

        iterId = 0
        epsilon = 0
        policy = initialPolicy
        V = initialV

        while iterId <= nIterations:

            V_iteration, useless_1, useless_2 = self.evaluatePolicyPartially(policy, V, nEvalIterations, tolerance)
            policy_iteration = self.extractPolicy(V_iteration)
            V_iteration = np.max(self.R + self.discount * (self.T.dot(V_iteration)), axis=0)

            epsilon = np.max(np.fabs(V_iteration-V))
            if(epsilon <= tolerance):
                V = V_iteration
                policy = policy_iteration
                return [policy, V, iterId, epsilon]

            V = V_iteration
            policy = policy_iteration
            iterId += 1

        return [policy, V, iterId, epsilon]


        