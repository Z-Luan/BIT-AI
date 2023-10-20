import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.test_env import EnvTest
from q4_schedule import LinearExploration, LinearSchedule
from core.deep_q_learning_torch import DQN

from configs.q6_nature import config


class NatureQN(DQN):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            (stride - 1) * img_height - stride + filter_size) // 2

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ################

        # nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        # stride步长默认为1, padding图像填充, 在上下左右均填充, 默认为0
        self.q_network = nn.Sequential( nn.Conv2d(n_channels * self.config.state_history , 32 , (8,8) , 4 , padding = ((4 - 1) * img_height - 4 + 8) // 2),
                                        nn.ReLU(),
                                        nn.Conv2d(32 , 64 , (4,4) , 2 , padding = ((2 - 1) * img_height - 2 + 4) // 2),
                                        nn.ReLU(),
                                        nn.Conv2d(64 , 64 , (3,3) , 1 , padding = (-1 + 3) // 2),
                                        nn.ReLU(),
                                        nn.Flatten(),
                                        # 推导可知 64 * img_height * img_width (前提img_height == img_width)
                                        nn.Linear(64 * img_height * img_width , 512),
                                        nn.ReLU(),
                                        nn.Linear(512 , num_actions)
        )   
        self.target_network = nn.Sequential( nn.Conv2d(n_channels * self.config.state_history , 32 , (8,8) , 4 , padding = ((4 - 1) * img_height - 4 + 8) // 2),
                                        nn.ReLU(),
                                        nn.Conv2d(32 , 64 , (4,4) , 2 , padding = ((2 - 1) * img_height - 2 + 4) // 2),
                                        nn.ReLU(),
                                        nn.Conv2d(64 , 64 , (3,3) , 1 , padding = (-1 + 3) // 2),
                                        nn.ReLU(),
                                        nn.Flatten(),
                                        nn.Linear(64 * img_height * img_width , 512),
                                        nn.ReLU(),
                                        nn.Linear(512 , num_actions)
                                        
        )
        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################
        state = state.permute(0,3,1,2)
        # print(f'Input shape after flattening = {input.shape}')
        if network == 'q_network':
            out = self.q_network(state)
        elif network == 'target_network':
            out = self.target_network(state)
        ##############################################################
        ######################## END YOUR CODE #######################
        return out

    def update_target(self):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights.

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.

        Hint:
            1. look up saving and loading pytorch models
        """

        ##############################################################
        ################### YOUR CODE HERE - 1-2 lines ###############
        self.target_network.load_state_dict(self.q_network.state_dict())
        ##############################################################
        ######################## END YOUR CODE #######################


    def calc_loss(self, q_values : Tensor, target_q_values : Tensor,
                    actions : Tensor, rewards: Tensor, done_mask: Tensor) -> Tensor:
        """
        Calculate the MSE loss of this step.
        The loss for an example is defined as:
            Q_samp(s) = r if done
                        = r + gamma * max_a' Q_target(s', a')
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')
            target_q_values: (torch tensor) shape = (batch_size, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')
            actions: (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
            rewards: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
            done_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state

        Hint:
            You may find the following functions useful
                - torch.max
                - torch.sum
                - torch.nn.functional.one_hot
                - torch.nn.functional.mse_loss
        """
        # you may need this variable
        num_actions = self.env.action_space.n
        gamma = self.config.gamma

        ##############################################################
        ##################### YOUR CODE HERE - 3-5 lines #############
        y_batch = rewards + (~done_mask) * gamma * torch.max(target_q_values , dim=1).values
        output_batch = torch.sum(q_values * torch.nn.functional.one_hot((actions).to(torch.int64) , num_classes = num_actions) , dim=1)
        loss = torch.nn.functional.mse_loss(y_batch , output_batch)
        return loss
        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer(self):
        """
        Set self.optimizer to be an Adam optimizer optimizing only the self.q_network
        parameters

        Hint:
            - Look up torch.optim.Adam
            - What are the input to the optimizer's constructor?
        """
        ##############################################################
        #################### YOUR CODE HERE - 1 line #############
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.00025)
        # self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-6)
        ##############################################################
        ######################## END YOUR CODE #######################


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
