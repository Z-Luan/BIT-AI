import gym
import numpy as np
import random
from copy import deepcopy

# github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

# github.com/google-research/google-research/blob/master/algae_dice/wrappers/normalize_action_wrapper.py
class NormalizeBoxActionWrapper(gym.ActionWrapper):
  """Rescale the action space of the environment."""

  def __init__(self, env):
    if not isinstance(env.action_space, gym.spaces.Box):
      raise ValueError('env %s does not use spaces.Box.' % str(env))
    super(NormalizeBoxActionWrapper, self).__init__(env)
    
  def action(self, action):
    # rescale the action
    low, high = self.env.action_space.low, self.env.action_space.high
    scaled_action = low + (action + 1.0) * (high - low) / 2.0
    scaled_action = np.clip(scaled_action, low, high)

    return scaled_action

  def reverse_action(self, scaled_action):
    low, high = self.env.action_space.low, self.env.action_space.high
    action = (scaled_action - low) * 2.0 / (high - low) - 1.0
    return action

# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render = False):
    states, actions, rewards = [], [], []
    states.append(env.reset()[0])
    done = False
    if render: env.render()
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done, trunc, info = env.step(action)
        done = done | trunc
        if render: env.render()
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards

# Play an episode according to a given policy and train
# env: environment
# policy: function(env, state)
# train: training function
# render: whether to render the episode or not (默认 False)
def play_episode_train(env, policy, train, render = False):
    states, actions, rewards = [], [], []
    states.append(env.reset()[0])
    done = False
    if render: env.render()
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done, trunc, info = env.step(action)
        done = done | trunc
        if render: env.render()
        states.append(obs)
        rewards.append(reward)
        if train: train(env, states, actions, rewards)
    return states, actions, rewards

# Play an episode according to a given policy and add 
# to a replay buffer
# env: environment
# policy: function(env, state)
def play_episode_rb(env, policy, buf):
    states, actions, rewards = [], [], []
    states.append(env.reset()[0])
    done = False
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, done, trunc, info = env.step(action)
        done = done | trunc
        buf.add(states[-1], action, reward, obs, done)
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards


# Toy Maze Environment
# Slide 9
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring20/slides/cs885-lecture3b.pdf
class ToyMaze(gym.Env):
    def __init__(self, H, W, rewards, obstacles, done, slip = 0.0):
        self.H = H
        self.W = W
        # Can't go to obstacles
        self.positions = [[i, j] for i in range(H) \
            for j in range(W) if [i, j] not in obstacles]
        self.position = None
        self.rewards = rewards
        self.obstacles = obstacles
        self.action_map = {
            'n': (0, 0),
            'r': (0, 1),
            'l': (0, -1),
            'u': (-1, 0),
            'd': (1, 0),
        }
        self.done = done
        self.slip = slip
        self.reset()
    def reset(self):
        self.position = random.choice(deepcopy(self.positions))
        return deepcopy(self.position)
    def transition_function(self, action):
        assert(action in self.action_map.keys())
        action_probabilities = deepcopy(self.action_map)
        check_valid_position1 = lambda i, j: [i, j] not in self.obstacles
        check_valid_position2 = lambda i, j: (0 <= i < self.H) and (0 <= j < self.W)
        check_valid_position = lambda i, j: check_valid_position1(i, j) and check_valid_position2(i, j)
        valid_positions = []
        for a in self.action_map.keys():
            ai, aj = self.action_map[a]
            if not check_valid_position(self.position[0]+ai, self.position[1]+aj) or \
                [self.position[0], self.position[1]] in self.done:
                action_probabilities[a] = 0.
            else:
                valid_positions += [a]
        if [self.position[0], self.position[1]] in self.done:
            # print('done')
            action_probabilities['n'] = 1.0
        elif action in valid_positions:
            # print('action valid')
            action_probabilities[action] = 1-self.slip
            unassigned = 0
            for a in self.action_map.keys():
                if type(action_probabilities[a]) == tuple:
                    unassigned += 1
            for a in self.action_map.keys():
                if type(action_probabilities[a]) == tuple:
                    action_probabilities[a] = self.slip/unassigned
        else:
            # print('action invalid')
            unassigned = 0
            for a in self.action_map.keys():
                if type(action_probabilities[a]) == tuple:
                    unassigned += 1
            for a in self.action_map.keys():
                if type(action_probabilities[a]) == tuple:
                    action_probabilities[a] = 1.0/unassigned
        return action_probabilities
    def step(self, action):
        assert(action in self.action_map.keys())
        if np.random.rand() <= self.slip:
            other_actions = list(set(list(self.action_map.keys()))-set([action]))
            action = random.choice(other_actions)
        ai, aj = self.action_map[action]
        if self.position in self.done:
            return deepcopy(self.position), self.rewards[self.position[0]][self.position[1]], \
                self.position in self.done, {}
        if [self.position[0]+ai, self.position[1]+aj] in self.obstacles:
            return deepcopy(self.position), self.rewards[self.position[0]][self.position[1]], \
                self.position in self.done, {}
        if not (0 <= self.position[0]+ai < self.H):
            return deepcopy(self.position), self.rewards[self.position[0]][self.position[1]], \
                self.position in self.done, {}
        if not (0 <= self.position[1]+aj < self.W):
            return deepcopy(self.position), self.rewards[self.position[0]][self.position[1]], \
                self.position in self.done, {}
        # Reward is based on current state, not the next state
        ret = self.rewards[self.position[0]][self.position[1]]
        retdone = self.position in self.done
        self.position[0] += ai
        self.position[1] += aj
        return deepcopy(self.position), ret, retdone, {}
    def render(self, mode=None):
        board = [['[ ]' for j in range(self.W)] for i in range(self.H)]
        for obstacle in self.obstacles:
            board[obstacle[0]][obstacle[1]] = '[O]'
        board[self.position[0]][self.position[1]] = '[P]'
        print("\n".join(["".join(line) for line in board]))
        print("")