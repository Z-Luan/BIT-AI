import collections
import numpy as np
import random
import torch

# Replay buffer
class ReplayBuffer:
    
    # create replay buffer of size N
    def __init__(self, N):
        self.buf = collections.deque(maxlen = N)
    
    # add: add a transition (s, a, r, s2, d)
    def add(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    
    # sample: return minibatch of size n
    def sample(self, n, t):
        minibatch = random.sample(self.buf, n)
        S, A, R, S2, D = [], [], [], [], []
        
        for mb in minibatch:
            s, a, r, s2, d = mb
            S += [s]; A += [a]; R += [r]; S2 += [s2]; D += [d]

        if type(A[0]) == int:
            return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D)
        elif type(A[0]) == float:
            return t.f(S), t.f(A), t.f(R), t.f(S2), t.i(D)
        else:
            return t.f(S), torch.stack(A), t.f(R), t.f(S2), t.i(D)
    def clear(self):
        self.buf = collections.deque(maxlen = N)
        
# Replay Buffer with states, actions, returns, log_probs
class ReplayBufferSARLP:

    # create replay buffer of size N
    def __init__(self, N, OBS_N, t):
        self.S = t.f(torch.zeros((N, OBS_N)))
        self.A = t.l(torch.zeros((N)))
        self.Ret = t.f(torch.zeros((N)))
        self.LP = t.f(torch.zeros((N)))
        self.i = 0
        self.N = N
        self.t = t
        self.filled = 0
    
    # add states, actions, returns, log_probs
    def add(self, states, actions, returns, log_probs):
        M = states.shape[0]
        self.filled = min(self.filled+M, self.N)
        assert(M <= self.N)
        for j in range(M):
            self.S[self.i] = self.t.f(states[j, :])
            self.A[self.i] = self.t.l(actions[j])
            self.Ret[self.i] = self.t.f(returns[j])
            self.LP[self.i] = self.t.f(log_probs[j])
            self.i = (self.i + 1) % self.N
    
    # sample: return minibatch of size n
    def sample(self, n):
        minibatch = random.sample(range(self.filled), n)
        S, A, Ret, LP = [], [], [], []
        
        for mbi in minibatch:
            s, a, ret, lp = self.S[mbi], self.A[mbi], self.Ret[mbi], self.LP[mbi]
            S += [s]; A += [a]; Ret += [ret]; LP += [lp]

        return torch.stack(S), torch.stack(A), torch.stack(Ret), torch.stack(LP)


# Replay Buffer with states, actions, returns, log_probs, probs
class ReplayBufferSARLPP:

    # create replay buffer of size N
    def __init__(self, N, OBS_N, ACT_N, t):
        self.S = t.f(torch.zeros((N, OBS_N)))
        self.A = t.l(torch.zeros((N)))
        self.Ret = t.f(torch.zeros((N)))
        self.LP = t.f(torch.zeros((N, ACT_N)))
        self.P = t.f(torch.zeros((N, ACT_N)))
        self.i = 0
        self.N = N
        self.t = t
        self.filled = 0
    
    # add states, actions, returns, log_probs, probs
    def add(self, states, actions, returns, log_probs, probs):
        M = states.shape[0]
        self.filled = min(self.filled+M, self.N)
        assert(M <= self.N)
        for j in range(M):
            self.S[self.i] = self.t.f(states[j, :])
            self.A[self.i] = self.t.l(actions[j])
            self.Ret[self.i] = self.t.f(returns[j])
            self.LP[self.i] = self.t.f(log_probs[j])
            self.P[self.i] = self.t.f(probs[j])
            self.i = (self.i + 1) % self.N
    
    # sample: return minibatch of size n
    def sample(self, n):
        minibatch = random.sample(range(self.filled), n)
        S, A, Ret, LP, P = [], [], [], [], []
        
        for mbi in minibatch:
            s, a, ret, lp, p = self.S[mbi], self.A[mbi], self.Ret[mbi], self.LP[mbi], self.P[mbi]
            S += [s]; A += [a]; Ret += [ret]; LP += [lp]; P += [p]

        return torch.stack(S), torch.stack(A), torch.stack(Ret), torch.stack(LP), torch.stack(P)