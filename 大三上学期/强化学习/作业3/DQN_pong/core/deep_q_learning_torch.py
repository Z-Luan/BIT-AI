import torch
import numpy as np
import torch.nn as nn

from typing import Tuple
from pathlib import Path
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from core.q_learning import QN

class DQN(QN):

    def __init__(self, env, config, logger=None):
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        print(f'Running model on device {self.device}')
        super().__init__(env, config, logger)
        self.summary_writer = SummaryWriter(self.config.output_path, max_queue=1e5)

    """
    Abstract class for Deep Q Learning
    """
    def initialize_models(self):
        """ Define the modules needed for the module to work."""
        raise NotImplementedError


    def get_q_values(self, state: Tensor, network: str) -> Tensor:
        """
        Input:
            state: A tensor of shape (batch_size, img height, img width, nchannels x config.state_history)

        Output:
            output: A tensor of shape (batch_size, num_actions)
        """
        raise NotImplementedError


    def update_target(self) -> None:
        """
        Update_target_op will be called periodically 
        to copy Q network to target Q network
    
        Args:
            q_scope: name of the scope of variables for q
            target_q_scope: name of the scope of variables for the target
                network
        """
        raise NotImplementedError


    def calc_loss(self, q_values : Tensor, target_q_values : Tensor, 
                    actions : Tensor, rewards: Tensor, done_mask: Tensor) -> Tensor:
        """
        Set (Q_target - Q)^2
        """
        raise NotImplementedError


    def add_optimizer(self) -> Optimizer:
        """
        Set training op wrt to loss for variable in scope
        """
        raise NotImplementedError


    def process_state(self, state : Tensor) -> Tensor:
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = state.float()
        state /= self.config.high

        return state


    def build(self):
        """
        Build model by adding all necessary variables
        """
        self.initialize_models()
        if hasattr(self.config, 'load_path'):
            print('Loading parameters from file:', self.config.load_path)
            load_path = Path(self.config.load_path)
            assert load_path.is_file(), f'Provided load_path ({load_path}) does not exist'
            self.q_network.load_state_dict(torch.load(load_path, map_location='cpu'))
            print('Load successful!')
        else:
            print('Initializing parameters randomly')
            def init_weights(m):
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight, gain=2 ** (1. / 2))
                if hasattr(m, 'bias'):
                    nn.init.zeros_(m.bias)
            self.q_network.apply(init_weights)
        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)
        self.add_optimizer()


    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # synchronise q and target_q networks
        assert self.q_network is not None and self.target_network is not None, \
            'WARNING: Networks not initialized. Check initialize_models'
        self.update_target()

       
    def add_summary(self, latest_loss, latest_total_norm, t):
        """
        Tensorboard stuff
        """
        self.summary_writer.add_scalar('loss', latest_loss, t)
        self.summary_writer.add_scalar('grad_norm', latest_total_norm, t)
        self.summary_writer.add_scalar('Avg_Reward', self.avg_reward, t)
        self.summary_writer.add_scalar('Max_Reward', self.max_reward, t)
        self.summary_writer.add_scalar('Std_Reward', self.std_reward, t)
        self.summary_writer.add_scalar('Avg_Q', self.avg_q, t)
        self.summary_writer.add_scalar('Max_Q', self.max_q, t)
        self.summary_writer.add_scalar('Std_Q', self.std_q, t)
        self.summary_writer.add_scalar('Eval_Reward', self.eval_reward, t)


    def save(self):
        """
        Saves session
        """
        # if not os.path.exists(self.config.model_output):
        #     os.makedirs(self.config.model_output)
        torch.save(self.q_network.state_dict(), self.config.model_output)
        # self.saver.save(self.sess, self.config.model_output)


    def get_best_action(self, state: Tensor) -> Tuple[int, np.ndarray]:
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
            s = self.process_state(s)
            action_values = self.get_q_values(s, 'q_network').squeeze().to('cpu').tolist()
        action = np.argmax(action_values)
        return action, action_values


    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """
        self.timer.start('update_step/replay_buffer.sample')
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)
        self.timer.end('update_step/replay_buffer.sample')
        
        assert self.q_network is not None and self.target_network is not None, \
            'WARNING: Networks not initialized. Check initialize_models'
        assert self.optimizer is not None, \
            'WARNING: Optimizer not initialized. Check add_optimizer'

        # Convert to Tensor and move to correct device
        self.timer.start('update_step/converting_tensors')
        s_batch = torch.tensor(s_batch, dtype=torch.uint8, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.uint8, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float, device=self.device)
        sp_batch = torch.tensor(sp_batch, dtype=torch.uint8, device=self.device)
        done_mask_batch = torch.tensor(done_mask_batch, dtype=torch.bool, device=self.device)
        self.timer.end('update_step/converting_tensors')

        # Reset Optimizer
        self.timer.start('update_step/zero_grad')
        self.optimizer.zero_grad()
        self.timer.end('update_step/zero_grad')

        # Run a forward pass
        self.timer.start('update_step/forward_pass_q')
        s = self.process_state(s_batch)
        q_values = self.get_q_values(s, 'q_network')
        self.timer.end('update_step/forward_pass_q')

        self.timer.start('update_step/forward_pass_target')
        with torch.no_grad():
            sp = self.process_state(sp_batch)
            target_q_values = self.get_q_values(sp, 'target_network')
        self.timer.end('update_step/forward_pass_target')

        self.timer.start('update_step/loss_calc')
        loss = self.calc_loss(q_values, target_q_values, 
            a_batch, r_batch, done_mask_batch)
        self.timer.end('update_step/loss_calc')
        self.timer.start('update_step/loss_backward')
        loss.backward()
        self.timer.end('update_step/loss_backward')

        # Clip norm
        self.timer.start('update_step/grad_clip')
        total_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.clip_val)
        self.timer.end('update_step/grad_clip')

        # Update parameters with optimizer
        self.timer.start('update_step/optimizer')
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        self.timer.end('update_step/optimizer')
        return loss.item(), total_norm.item()


    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.update_target()

