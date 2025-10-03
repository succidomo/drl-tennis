import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_out = layer.weight.data.size(0)  # match reference behavior
    lim = 1. / np.sqrt(fan_out)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        #self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        #for m in [self.fc1, self.fc2, self.fc3]:
        #    nn.init.zeros_(m.bias)

        #self.bn1.reset_parameters()
        #self.bn2.reset_parameters()


    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = F.leaky_relu(self.fc1(state))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        #x = self.bn2(x)
        return torch.tanh(self.fc3(x))
    
# model.py
class Critic(nn.Module):
    def __init__(self, full_state_size, seed, hidden1=256, hidden2=256):
        super().__init__()
        torch.manual_seed(seed)
        self.fcs1 = nn.Linear(full_state_size, hidden1)  # Input: full obs_all (e.g., 48)
        self.fc2 = nn.Linear(hidden1, hidden2)  # Cat hidden + full act_all (256+4=260)
        #self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        # for m in [self.fcs1, self.fc2, self.fc3]:
        #     nn.init.zeros_(m.bias)
        # self.bn1.reset_parameters()
        # self.bn2.reset_parameters()

    def forward(self, obs_all, act_all):
        xs = torch.cat((obs_all, act_all), dim=1)
        x = F.leaky_relu(self.fcs1(xs))  # Process full obs first
        x = self.bn1(x)
        #x = torch.cat([x, act_all], dim=1)   # Now cat actions to hidden
        x = F.leaky_relu(self.fc2(x))
        #x = self.bn2(x)
        return self.fc3(x)
    

class DDPG_Models():

    def __init__(self, num_agents, state_size=24, action_size=2, seed=0):
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        input_size = (state_size+action_size)*num_agents
        self.critic_local = Critic(input_size, seed).to(device)
        self.critic_target = Critic(input_size, seed).to(device)


