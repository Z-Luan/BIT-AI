import gym
import subprocess
from pathlib import Path
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from q4_schedule import LinearExploration, LinearSchedule
from q6_nature_torch import NatureQN

from configs.q7_train_atari_nature import config

"""
Use deep Q network for the Atari game. Please report the final result.
Feel free to change the configurations (in the configs/ folder). 
If so, please report your hyperparameters.

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.

To launch tensorboard, open a Terminal window and run 
tensorboard --logdir=results/
Then, connect remotely to 
address-ip-of-the-server:6006 
6006 is the default port used by tensorboard.
"""
if __name__ == '__main__':
    # make env
    # atari-py-0.2.6
    # gym-0.19.0
    # https://www.bilibili.com/read/cv17216467/
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # Check weights
    load_path = Path(config.load_path)
    if load_path.is_file():
        print(f'File {load_path} exists')
    else:
        print(f'please download File {load_path}')

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
