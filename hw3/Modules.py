import numpy as np
import torch
import torch.nn as nn
import random

from collections import deque, namedtuple


class NormalModule(nn.Module):
    def __init__(self, inp, out, activation=nn.Tanh):
        super().__init__()
        self.m = nn.Linear(inp, out)
        log_std = -0.5 * np.ones(out, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.act1 = activation

    def forward(self, inputs):
        mout = self.m(inputs)
        vout = torch.exp(self.log_std)
        return mout, vout


'''* Create and test an experience replay buffer with a random policy, which is the 
Gaussian distribution with arbitrary (randomly initialized) weights of the policy feed-forward network,
receiving state, s, and returning the mean, mu(s) and the log_std, log_stg(s) 
(natural logarithm of the standard deviation) of actions.  As mentioned above, you can use 
a state-independent standard variance.'''
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ExperienceReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the Policy Neural Network as a simple feedforward NN with 2 hidden layers
# for the policy function
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.z_mu = nn.Linear(hidden_size, output_size) # a1
        self.z_var = nn.Linear(hidden_size, output_size)
        #nn.init.normal_(self.z_var.weight, mean=0, std=0.1)

    def forward(self, x):
        
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        #x = nn.functional.softmax(self.fc3(x))
        
        mean = self.z_mu(x)
        log_var = self.z_var(x)
        '''x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        mu = self.z_mu(x)'''
        
        #log_std = self.z_var(x)
        
        return mean, log_var
        

    
class CriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        return x