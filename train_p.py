# -*- coding: utf-8 -*-
"""
@author: Zhang Tianming
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import shutil
import random
import numpy as np
import time
import cPickle as pickle
from collections import defaultdict, deque
from game import Board, Game
from policy_value_net import PolicyValueNet
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphazero import MCTSPlayer
import multiprocessing
import threading
import os
from multiprocessing import Pool


def get_equi_data(play_data, board_height, board_width):
    """
    augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]"""
    extend_data = []
    for state, mcts_porb, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(board_height, board_width)), i)
            extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
    return extend_data


def collect_selfplay_data(data_queue, game,
                          board_width, board_height,
                          c_puct, n_playout, temp,
                          model_file, n_games=1):
    """collect self-play data for training"""
    policy_value_net = PolicyValueNet(board_width, board_height)
    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=c_puct,
                             n_playout=n_playout, is_selfplay=1)
    while True:
        for i in range(n_games):
            winner, play_data = game.start_self_play(mcts_player, temp=temp)
            episode_len = len(play_data)
            # augment the data
            play_data = get_equi_data(play_data, board_width, board_height)
            for data in play_data:
                data_queue.put(data)
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file)
        else:
            checkpoint = None
        policy_value_net = PolicyValueNet(board_width, board_height, checkpoint)
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=c_puct,
                                 n_playout=n_playout, is_selfplay=1)


def policy_evaluate(win_queue, job_queue, game, role,
                    board_width, board_height,
                    c_puct, n_playout, pure_mcts_playout_num,
                    model_file):
    """
    Evaluate the trained policy by playing games against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    while True:
        while job_queue.empty():
            time.sleep(1)
        job_queue.get()
        checkpoint = torch.load(model_file)
        policy_value_net = PolicyValueNet(board_width, board_height, checkpoint)
        current_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=c_puct,
                                         n_playout=n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=pure_mcts_playout_num)
        winner = game.start_play(current_mcts_player, pure_mcts_player, start_player=role, is_shown=0)
        win_queue.put(winner)


class TrainPipeline():
    def __init__(self):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 5e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.025
        self.check_freq = 50
        self.game_batch_num = 1500
        self.n_games_eval = 10
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.model_file = 'checkpoint.pth.tar'
        self.best_model_name = 'checkpoint_best.pth.tar'
        # start training from a given policy-value net
        #        policy_param = pickle.load(open('current_policy.model', 'rb'))
        #        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, net_params = policy_param)
        # start training from a new policy-value net

    def init_model(self):
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                      n_playout=self.n_playout, is_selfplay=1)

    def policy_update(self):
        """update the policy-value net"""
        t1 = time.time()
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch])
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        winner_batch = np.array([data[2] for data in mini_batch])

        state_batch_v = Variable(torch.Tensor(state_batch.copy()).type(torch.FloatTensor).cuda())
        mcts_probs_batch_v = Variable(torch.Tensor(mcts_probs_batch.copy()).type(torch.FloatTensor).cuda())
        winner_batch_v = Variable(torch.Tensor(winner_batch.copy()).type(torch.FloatTensor).cuda())
        old_probs, old_v = self.policy_value_net.policy_value_model(state_batch_v)
        old_probs, old_v = old_probs.data.cpu().numpy(), old_v.data.cpu().numpy()
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch_v, mcts_probs_batch_v, winner_batch_v,
                                                             self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value_model(state_batch_v)
            new_probs, new_v = new_probs.data.cpu().numpy(), new_v.data.cpu().numpy()
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))
        t2 = time.time()
        print(
            "kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f},time_used:{:.3f}".format(
                kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new, t2 - t1))
        return loss, entropy

    def policy_evaluate(self):
        manager = multiprocessing.Manager()
        self.win_queue = manager.Queue(maxsize=self.n_games_eval)
        self.job_queue = manager.Queue(maxsize=self.n_games_eval)
        procs = []
        for i in range(4):
            start_role = i % 2
            args = (self.win_queue, self.job_queue,
                    self.game, start_role,
                    self.board_width, self.board_height,
                    self.c_puct, self.n_playout, self.pure_mcts_playout_num,
                    self.model_file,)
            proc = multiprocessing.Process(target=policy_evaluate, args=args)
            procs.append(proc)
            proc.start()
        self.eval_procs = procs

    def get_win_ratio(self):
        for i in range(self.n_games_eval):
            self.job_queue.put(1)
        win_cnt = defaultdict(int)
        for i in range(self.n_games_eval):
            winner = self.win_queue.get()
            win_cnt[winner] += 1
            print(i)
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / self.n_games_eval
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_playout_num, win_cnt[1], win_cnt[2],
                                                                  win_cnt[-1]))
        return win_ratio

    def collect_selfplay_data(self):
        """run the training pipeline"""
        manager = multiprocessing.Manager()
        self.data_queue = manager.Queue(maxsize=5120)
        NUM_PROCESS = 2
        procs = []
        for i in range(NUM_PROCESS):
            proc = multiprocessing.Process(target=collect_selfplay_data,
                                           args=(self.data_queue, self.game,
                                                 self.board_width, self.board_height,
                                                 self.c_puct, self.n_playout, self.temp,
                                                 self.model_file, 1,))
            procs.append(proc)
            proc.start()
        self.collect_procs = procs

    def train(self):
        try:
            for i in range(self.game_batch_num):
                t1 = time.time()
                while True:
                    while self.data_queue.empty():
                        time.sleep(1)
                    item = self.data_queue.get()
                    self.data_buffer.append(item)
                    if len(self.data_buffer) > self.batch_size:
                        break
                t2 = time.time()
                print("batch i:{}, data_buffer_size:{},time_used:{:.3f}".format(i + 1, len(self.data_buffer), t2 - t1))
                loss, entropy = self.policy_update()
                state = {'state_dict': self.policy_value_net.policy_value_model.state_dict(),
                         'optim_dict': self.policy_value_net.optimizer.state_dict(),
                         'loss': loss,
                         'entropy': entropy}
                torch.save(state, self.model_file)
                # check the performance of the current model，and save the model params
                if (i + 1) % self.check_freq == 0:
                    t1 = time.time()
                    print("current self-play batch: {}, start to evaluate...".format(i + 1))
                    win_ratio = self.get_win_ratio()
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                        torch.save(state, self.best_model_name)
                    t2 = time.time()
                    print("current self-play batch: {}, end to evaluate...,time_used:{:.3f}".format(i + 1, t2 - t1))
        except KeyboardInterrupt:
            print('\n\rquit')

    def release(self):
        for proc in self.collect_procs:
            proc.terminate()
            proc.join()
        for proc in self.eval_procs:
            proc.terminate()
            proc.join()


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.collect_selfplay_data()
    training_pipeline.policy_evaluate()
    training_pipeline.init_model()
    print('start training')
    training_pipeline.train()
    training_pipeline.release()
