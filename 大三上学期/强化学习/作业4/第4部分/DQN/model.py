import torch
import torch.nn as nn
import torch.nn.functional as F

from config import gamma, sequence_length, device
class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        # The inputs are two integers giving the dimensions of the inputs and outputs respectively. 
        # The input dimension is the state dimention and the output dimension is the action dimension.
        # This constructor function initializes the network by creating the different layers. 
        
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs * sequence_length, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        # The variable x denotes the input to the network. 
        # The function returns the q value for the given input. 

        x = x.view(-1, self.num_inputs * sequence_length)
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):

        # The online_net is the variable that represents the first (current) Q network.
        # The target_net is the variable that represents the second (target) Q network.
        # The optimizer is Adam. 
        # Batch represents a mini-batch of memory. 

        # This function takes in a mini-batch of memory, calculates the loss and trains the online network. Target network is not trained using back prop. 
        # Return the value of loss for logging purposes (optional).


        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        actions = actions.to(device)
        rewards = torch.Tensor(batch.reward)
        rewards = rewards.to(device)
        masks = torch.Tensor(batch.mask)
        masks = masks.to(device)

        pred = online_net(states)
        next_pred = target_net(next_states)
        pred = torch.sum(pred.mul(actions), dim=1)
        target = rewards + masks * gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss 

    def get_action(self, input_var):
        # This function obtains the action from the DQN, by calculating the Q values for the given input variable.

        qvalue = self.forward(input_var)
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]
