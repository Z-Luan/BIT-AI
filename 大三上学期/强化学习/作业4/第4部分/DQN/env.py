import gym
from config import env_name




class partial_env_class():
    def __init__(self):
        self.env = gym.make(env_name)
    
    def set_seed(self, seed):
        self.env.seed(seed)

    def get_random_action(self): 
        return self.env.action_space.sample()

    def state_to_partial_observability(self, state):
        state = state[[0, 2]]
        return state
    
    def num_actions(self):
        return self.env.action_space.n

    
    def num_states(self):
        number_of_states = 2
        return number_of_states

    
    def reset(self): 
        state = self.env.reset()
        state = self.state_to_partial_observability(state)
        return state 

    
    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.state_to_partial_observability(next_state)
        return next_state, reward, done





