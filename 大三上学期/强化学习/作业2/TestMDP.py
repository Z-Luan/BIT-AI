from MDP import *

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
print("价值迭代测试：")
print("V=",V,",epsilon=",epsilon,"nIterations=",nIterations)
print("")
policy = mdp.extractPolicy(V)
print("价值迭代后，提取到的策略：")
print("policy=",policy)
print("")
V = mdp.evaluatePolicy(np.array([1,0,1,0]))
print("策略为[1,0,1,0]时的评估结果：")
print("V=",V)
print("")
[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print("策略迭代的结果：")
print("policy=",policy,"V=",V,"iterId",iterId)
print("")
[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,0,1,0]),np.array([0,10,0,13]))
print("策略为[1,0,1,0]，值函数为[0,10,0,13]，部分策略评估的结果：")
print("V=",V,"iterId=",iterId,"epsilon=",epsilon)
print("")
[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]))
print("修改后的策略迭代结果：")
print("policy=",policy,"V=",V,"iterId=",iterId,"tolerance=",tolerance)