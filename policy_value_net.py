# -*- coding: utf-8 -*-
"""
@author: Zhang Tianming
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

from game import *


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, use_batchnorm=False, bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)  # verify bias false
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)
        # self.act = nn.ReLU(inplace=False)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.act(x)
        # x = self.act(x) * x
        return x


class PolicyValueBackBoneNet(nn.Module):
    def __init__(self, num_actions, checkpoint=None):
        super(PolicyValueBackBoneNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = BasicConv2d(4, 32, 3, 1, 1, use_batchnorm=False, bias=False)
        self.conv2 = BasicConv2d(32, 64, 3, 1, 1, use_batchnorm=False, bias=False)
        self.conv3 = BasicConv2d(64, 128, 3, 1, 1, use_batchnorm=False, bias=False)
        self.action_head_conv1 = BasicConv2d(128, 4, 1, 1, 0, use_batchnorm=False, bias=False)
        self.action_head = nn.Linear(4 * self.num_actions, self.num_actions)
        self.value_head_conv1 = BasicConv2d(128, 2, 1, 1, 0, use_batchnorm=False, bias=False)
        self.value_head_fc1 = nn.Linear(2 * self.num_actions, 64)
        self.value_head = nn.Linear(64, 1)

        if checkpoint is not None:
            if checkpoint['state_dict'].keys()[0].startswith('module.') and \
                    checkpoint['state_dict'].keys()[-1].startswith('module.'):
                checkpoint['state_dict'] = dict((k[7:], v) for k, v in checkpoint['state_dict'].items())
            model_dict = self.state_dict()
            model_dict.update(checkpoint['state_dict'])
            self.load_state_dict(model_dict)

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        action_head_net = self.action_head_conv1(net)
        action_head_net = action_head_net.view(action_head_net.size(0), -1)
        action_scores = self.action_head(action_head_net)
        value_head_net = self.value_head_conv1(net)
        value_head_net = value_head_net.view(value_head_net.size(0), -1)
        value_head_net = self.value_head_fc1(value_head_net)
        state_values = self.value_head(value_head_net)
        return F.softmax(action_scores, dim=1), F.tanh(state_values)


class PolicyValueNet(object):
    """policy-value network """

    def __init__(self, board_width, board_height, checkpoint=None):
        self.board_width = board_width
        self.board_height = board_height
        self.checkpoint = checkpoint
        self.l2_const = 1e-4  # coef of l2 penalty
        self.create_policy_value_net()
        self.optimizer = optim.Adam(self.policy_value_model.parameters(), lr=3e-2)

    def create_policy_value_net(self):
        self.policy_value_model = PolicyValueBackBoneNet(self.board_height * self.board_width, self.checkpoint)
        self.policy_value_model.cuda()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()
        self.policy_value_model.eval()
        current_state = current_state.reshape(-1, 4, self.board_width, self.board_height)
        current_state = Variable(torch.Tensor(current_state.copy()).type(torch.FloatTensor).cuda())
        act_probs, value = self.policy_value_model(current_state)
        act_probs, value = act_probs.data.cpu().numpy(), value.data.cpu().numpy()
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def train_step(self, state_input, mcts_probs, winner, learning_rate):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """
        self.policy_value_model.train()
        for idx, group in enumerate(self.optimizer.param_groups):
            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1
            group['lr'] = learning_rate
        act_probs, value = self.policy_value_model(state_input)
        value_losses = F.smooth_l1_loss(value, winner)
        policy_losses = (-act_probs.log() * mcts_probs).sum(dim=-1)
        self.optimizer.zero_grad()
        loss = policy_losses.mean() + value_losses.mean()
        loss.backward()
        self.optimizer.step()
        entropy = (-act_probs.log() * act_probs).sum(dim=-1)
        entropy = entropy.mean()
        return loss.data[0], entropy.data[0]


if __name__ == '__main__':
    pvnet = PolicyValueNet(8, 8)
    b1 = Board(height=8, width=8, n_in_row=5)
    b1.init_board()
    print pvnet.policy_value_fn(b1)

    # bs1 = torch.from_numpy(np.random.rand(3,4,8,8)).type(torch.FloatTensor).cuda()
    # print pvnet.policy_value_model(Variable(bs1))
