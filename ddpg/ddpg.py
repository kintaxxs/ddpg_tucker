import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
def MSELoss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Actor(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        self.emb = nn.Embedding(512,32)
        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs*32, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)


    def forward(self, inputs):
        x = inputs
        x = self.emb(x)
        x = self.bn0(x)
        x = x.view(x.size(0),4*32)
        x = F.sigmoid(self.linear1(x))
        x = F.sigmoid(self.linear2(x))

        mu = F.sigmoid(self.mu(x))
        return mu

    
class Critic(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        self.emb = nn.Embedding(512,32)
        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0) 

        self.linear1 = nn.Linear(num_inputs*32, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear_action = nn.Linear(num_outputs, hidden_size)
        self.bn_a = nn.BatchNorm1d(hidden_size)
        self.bn_a.weight.data.fill_(1)
        self.bn_a.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = inputs
        x = self.emb(x)
        x = self.bn0(x)
        x = x.view(x.size(0),4*32)
        x = F.sigmoid(self.linear1(x))
        a = F.sigmoid(self.linear_action(actions))
        x = torch.cat((x, a), 1)
        x = F.sigmoid(self.linear2(x))

        V = self.V(x)
        return V


class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, exploration=None):
        self.actor.eval()
        with torch.no_grad():
            st = np.asarray(state, dtype=int)
            st = torch.from_numpy(st)
            mu = self.actor((Variable(st)))
        self.actor.train()
        mu = mu.data
        if exploration is not None:
            mu += torch.Tensor(exploration.noise())

        return mu.clamp(0, 1)


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        with torch.no_grad():
            next_state_batch = Variable(torch.cat(batch.next_state))
        
        sb = np.asarray(state_batch.cpu().numpy(), dtype=int)
        sb = torch.from_numpy(sb)
        nsb = np.asarray(next_state_batch.cpu().numpy(), dtype=int)
        nsb = torch.from_numpy(nsb)
        next_action_batch = self.actor_target(nsb)
        next_state_action_values = self.critic_target(nsb, next_action_batch)

        reward_batch = torch.unsqueeze(reward_batch, 1)
        expected_state_action_batch = reward_batch + (self.gamma * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((sb), (action_batch))

        value_loss = MSELoss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((sb),self.actor((sb)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
