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
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, use_batchnorm=True, bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)  # verify bias false
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, planes, use_batchnorm=True):
        super(ResidualBlock, self).__init__()
        self.conv1_1 = BasicConv2d(planes, planes, 3, 1, 1, use_batchnorm=use_batchnorm, bias=False)
        self.conv1_2 = BasicConv2d(planes, planes, 3, 1, 1, use_batchnorm=use_batchnorm, bias=False)

    def forward(self, x):
        x = self.conv1_2(self.conv1_1(x)) + x
        return x


class PolicyValueBackBoneNet(nn.Module):
    def __init__(self, num_actions, feature_planes=4, checkpoint=None):
        super(PolicyValueBackBoneNet, self).__init__()
        self.feature_planes = feature_planes
        self.num_actions = num_actions

        conv1 = BasicConv2d(self.feature_planes, 256, 3, 1, 1)
        residuals = [ResidualBlock(256) for i in range(10)]
        self.seqs = nn.Sequential(*tuple([conv1] + residuals))

        '''
        conv1 = BasicConv2d(self.feature_planes, 32, 3, 1, 1)
        conv2 = BasicConv2d(32, 64, 3, 1, 1)
        conv3 = BasicConv2d(64, 128, 3, 1, 1)
        conv4 = BasicConv2d(128, 256, 3, 1, 1)
        self.seqs = nn.Sequential(conv1,conv2,conv3,conv4)
        '''

        self.action_head_conv1 = BasicConv2d(256, 2, 1, 1, 0, use_batchnorm=False, bias=False)
        self.action_head = nn.Linear(2 * self.num_actions, self.num_actions)
        self.value_head_conv1 = BasicConv2d(256, 1, 1, 1, 0, use_batchnorm=False, bias=False)
        self.value_head_fc1 = nn.Linear(self.num_actions, 256)
        self.value_head = nn.Linear(256, 1)
        self.resume(checkpoint)

    def resume(self, checkpoint):
        if checkpoint is not None:
            if checkpoint['state_dict'].keys()[0].startswith('module.') and \
                    checkpoint['state_dict'].keys()[-1].startswith('module.'):
                checkpoint['state_dict'] = dict((k[7:], v) for k, v in checkpoint['state_dict'].items())
            model_dict = self.state_dict()
            model_dict.update(checkpoint['state_dict'])
            self.load_state_dict(model_dict)

    def forward(self, x):
        net = self.seqs(x)
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

    def __init__(self, board_width, board_height, feature_planes=4, mode='train', checkpoint=None):
        self.board_width = board_width
        self.board_height = board_height
        self.feature_planes = feature_planes
        self.checkpoint = checkpoint
        self.mode = mode
        self.l2_const = 1e-4  # coef of l2 penalty
        self.create_policy_value_net()
        self.optimizer = self.create_optimizer(self.policy_value_model,'sgd',lr=3e-2,weight_decay=self.l2_const)
        #self.optimizer = optim.Adam(self.policy_value_model.parameters(), lr=3e-2,weight_decay=self.l2_const)

    def create_optimizer(self,model, optimi_str, lr, weight_decay, args):
        # setup optimizer
        if optimi_str == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=0.9,
                                  weight_decay=weight_decay, nesterov=True)
        elif optimi_str == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr,
                                   weight_decay=weight_decay, betas=(args.beta1, 0.999))
        elif optimi_str == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(),
                                      lr=lr,
                                      lr_decay=args.lr_decay,
                                      weight_decay=weight_decay)
        elif optimi_str == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(),
                                      lr=lr,
                                      alpha=0.99,
                                      eps=1e-08,
                                      weight_decay=weight_decay,
                                      momentum=0, centered=False)
        return optimizer

    def create_policy_value_net(self):
        self.policy_value_model = PolicyValueBackBoneNet(self.board_height * self.board_width,
                                                         self.feature_planes, self.checkpoint)

        if self.mode == 'train':
            self.policy_value_model = torch.nn.DataParallel(self.policy_value_model).cuda()
            self.policy_value_model.train()
        else:
            self.policy_value_model.cuda()
            self.policy_value_model.eval()

    def resume(self, checkpoint):
        self.policy_value_model.resume(checkpoint)

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()
        current_state = current_state.reshape(-1, self.feature_planes, self.board_width, self.board_height)
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
        return act_probs, value, loss.data[0], entropy.data[0]


if __name__ == '__main__':
    pvnet = PolicyValueNet(8, 8, 4)
    b1 = Board(height=8, width=8, feature_planes=4, n_in_row=5)
    b1.init_board()
    print pvnet.policy_value_fn(b1)

    # bs1 = torch.from_numpy(np.random.rand(3,4,8,8)).type(torch.FloatTensor).cuda()
    # print pvnet.policy_value_model(Variable(bs1))
