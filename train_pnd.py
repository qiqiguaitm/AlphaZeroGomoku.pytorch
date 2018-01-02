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
from negamax import NegamaxPlayer
import argparse
from dist.client import *
from dist.data_server import *

DIST_DATA_URL = 'http://10.83.150.55:8000/'


# DIST_DATA_URL = 'http://10.93.189.54:8000/'


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


def collect_selfplay_data(pid, gpu_id, data_queue, data_queue_lock, game,
                          board_width, board_height, feature_planes,
                          c_puct, n_playout, temp,
                          model_file, n_games=1,
                          is_distributed=False, data_server_url=DIST_DATA_URL):
    """collect self-play data for training"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # time.sleep(int(pid) * 5)
    n_epoch = 0
    checkpoint = None
    while True:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        policy_value_net = PolicyValueNet(board_width, board_height, feature_planes, mode='eval', checkpoint=checkpoint)
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=c_puct,
                                 n_playout=n_playout, is_selfplay=1)
        if not is_distributed:
            while data_queue.qsize() > 512 * 20:
                time.sleep(1)
        for n_game in range(n_games):
            print('PID:%s,N_EPOCH:%s,N_GAME:%s start ...' % (pid, n_epoch, n_game))
            t1 = time.time()
            winner, play_data = game.start_self_play(mcts_player, temp=temp)
            t2 = time.time()
            print('PID:%s,N_EPOCH:%s,N_GAME:%s end, time_used:%s' % (pid, n_epoch, n_game, t2 - t1))
            # augment the data
            play_data = get_equi_data(play_data, board_width, board_height)
            print('PID:%s,N_EPOCH:%s,N_GAME:%s send data ....' % (pid, n_epoch, n_game))
            if is_distributed:
                upload_samples(data_server_url, play_data)
            else:
                data_queue_lock.acquire()
                for data in play_data:
                    data_queue.put(data)
                data_queue_lock.release()
            print('PID:%s,N_EPOCH:%s,N_GAME:%s send data end.' % (pid, n_epoch, n_game))
        n_epoch += 1
        while True:
            try:
                if is_distributed:
                    download(data_server_url, model_file, model_file)
                checkpoint = torch.load(model_file)
                break
            except:
                continue


def policy_evaluate(gpu_id, win_queue, job_queue, job_queue_lock, game, role,
                    board_width, board_height, feature_planes,
                    c_puct, n_playout, pure_mcts_playout_num,
                    model_file):
    """
    Evaluate the trained policy by playing games against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    while True:
        while job_queue.empty():
            time.sleep(1)
        search_depth = job_queue.get()
        checkpoint = torch.load(model_file)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        policy_value_net = PolicyValueNet(board_width, board_height, feature_planes, checkpoint=checkpoint)
        current_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=c_puct,
                                         n_playout=n_playout)
        #refer_player = MCTS_Pure(c_puct=5, n_playout=pure_mcts_playout_num)
        refer_player = NegamaxPlayer(cmd_path='negamax/build/renju',search_depth=search_depth)
        winner = game.start_play(current_mcts_player, refer_player, start_player=role, is_shown=0)
        job_queue_lock.acquire()
        win_queue.put(winner)
        job_queue_lock.release()


class TrainPipeline():
    def __init__(self):
        # params of the board and the game
        self.board_width = 11
        self.board_height = 11
        self.feature_planes = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height,
                           feature_planes=self.feature_planes, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 5e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        #self.n_playout = 400  # num of simulations for each move
        self.n_playout = 800  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 512 * 20
        self.batch_size = 512 * 4  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.025
        self.check_freq = 50
        self.game_batch_num = 150000
        self.n_games_eval = 10
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.negamax_search_depth = 2
        self.gpus = ['0', '1', '2', '3']
        # self.gpus = ['0']
        # self.gpus = [str(i) for i in range(int(torch.cuda.device_count()))]
        self.num_inst = 0
        self.model_file = 'checkpoint.pth.tar'
        self.best_model_file = 'checkpoint_best.pth.tar'
        self.manager = multiprocessing.Manager()

    def init_model(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.gpus)
        checkpoint = None
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file)
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, self.feature_planes,checkpoint=checkpoint)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                      n_playout=self.n_playout, is_selfplay=1)

    def policy_update(self):
        """update the policy-value net"""
        t1 = time.time()
        t11 = time.time()
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch])
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        winner_batch = np.array([data[2] for data in mini_batch])
        state_batch_v = Variable(torch.Tensor(state_batch.copy()).type(torch.FloatTensor).cuda())
        mcts_probs_batch_v = Variable(torch.Tensor(mcts_probs_batch.copy()).type(torch.FloatTensor).cuda())
        winner_batch_v = Variable(torch.Tensor(winner_batch.copy()).type(torch.FloatTensor).cuda())
        t12 = time.time()
        t21 = time.time()
        old_probs, old_v, loss, entropy = self.policy_value_net.train_step(state_batch_v, mcts_probs_batch_v,
                                                                           winner_batch_v,
                                                                           self.learn_rate * self.lr_multiplier)
        t22 = time.time()
        old_probs, old_v = old_probs.data.cpu().numpy(), old_v.data.cpu().numpy()
        for i in range(self.epochs - 1):
            new_probs, new_v, loss, entropy = self.policy_value_net.train_step(state_batch_v, mcts_probs_batch_v,
                                                                               winner_batch_v,
                                                                               self.learn_rate * self.lr_multiplier)
            new_probs, new_v = new_probs.data.cpu().numpy(), new_v.data.cpu().numpy()
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        new_probs, new_v = self.policy_value_net.policy_value_model(state_batch_v)
        new_probs, new_v = new_probs.data.cpu().numpy(), new_v.data.cpu().numpy()
        kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
           self.lr_multiplier *= 1.5
        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))
        t2 = time.time()
        print(
            "kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f},"
            "time_used_for_data:{:.3f},time_used_for_trainstep:{:.3f},time_used:{:.3f}".format(
                kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new, t12 - t11, t22 - t21,
                                                                                             t2 - t1))
        return loss, entropy

    def policy_evaluate(self):
        self.manager = multiprocessing.Manager()
        self.win_queue = self.manager.Queue(maxsize=self.n_games_eval)
        self.job_queue = self.manager.Queue(maxsize=self.n_games_eval)
        self.job_queue_lock = self.manager.Lock()
        procs = []
        NUM_PROCESS = self.n_games_eval
        for idx in range(NUM_PROCESS):
            start_role = idx % 2
            gpu_id = self.gpus[self.num_inst % len(self.gpus)]
            self.num_inst += 1
            args = (gpu_id, self.win_queue, self.job_queue, self.job_queue_lock,
                    self.game, start_role,
                    self.board_width, self.board_height, self.feature_planes,
                    self.c_puct, self.n_playout, self.pure_mcts_playout_num,
                    self.model_file)
            proc = multiprocessing.Process(target=policy_evaluate, args=args)
            procs.append(proc)
            proc.start()
        self.eval_procs = procs

    def get_win_ratio(self,search_depth=2):
        self.job_queue_lock.acquire()
        for i in range(self.n_games_eval):
            self.job_queue.put(search_depth)
        self.job_queue_lock.release()
        win_cnt = defaultdict(int)
        for i in range(self.n_games_eval):
            winner = self.win_queue.get()
            win_cnt[winner] += 1
            print(i)
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / self.n_games_eval
        print("search_depth:{}, win: {}, lose: {}, tie:{}".format(search_depth, win_cnt[1], win_cnt[2],
                                                                  win_cnt[-1]))
        return win_ratio

    def collect_selfplay_data(self, is_distributed=False, data_server_url=DIST_DATA_URL):
        """run the training pipeline"""
        self.data_queue = self.manager.Queue(maxsize=5120)
        self.data_queue_lock = self.manager.Lock()
        NUM_PROCESS = 24
        procs = []
        for idx in range(NUM_PROCESS):
            gpu_id = self.gpus[self.num_inst % len(self.gpus)]
            self.num_inst += 1
            proc = multiprocessing.Process(target=collect_selfplay_data,
                                           args=(idx, gpu_id, self.data_queue, self.data_queue_lock,
                                                 self.game, self.board_width, self.board_height, self.feature_planes,
                                                 self.c_puct, self.n_playout, self.temp,
                                                 self.model_file, 1,
                                                 is_distributed, data_server_url))
            procs.append(proc)
            proc.start()
        self.collect_procs = procs

    def eval(self, is_distributed=False, data_server_url=DIST_DATA_URL):
        # check the performance of the current model，and save the model params
        idx = 0
        while True:
            checkpoint = None
            while checkpoint is None:
                try:
                    if is_distributed:
                        download(data_server_url, self.model_file, self.model_file)
                    checkpoint = torch.load(self.model_file)
                    break
                except:
                    continue
            t1 = time.time()
            print("current self-play batch: {}, start to evaluate...".format(idx + 1))
            win_ratio = self.get_win_ratio(search_depth=self.negamax_search_depth)
            if win_ratio > self.best_win_ratio:
                print("New best policy!!!!!!!!")
                self.best_win_ratio = win_ratio
                if self.best_win_ratio == 1.0 and self.negamax_search_depth < 20 and self.negamax_search_depth != -1:
                    self.negamax_search_depth += 2
                    if self.negamax_search_depth >= 20:
                        self.negamax_search_depth = -1
                    self.best_win_ratio = 0.0
                shutil.copy(self.model_file, self.best_model_file)
            t2 = time.time()
            print("current self-play batch: {}, end to evaluate...,time_used:{:.3f}".format(idx + 1, t2 - t1))
            idx = idx + 1

    def train(self, is_distributed=False, data_server_url=DIST_DATA_URL):
        samples_holder = deque(maxlen=self.buffer_size * 300)
        try:
            for i in range(self.game_batch_num):
                t1 = time.time()
                cnt = 0
                if is_distributed:
                    while True:
                        samples = download_samples()
                        samples_holder.extend(samples)
                        while True:
                            try:
                                self.data_buffer.append(samples_holder.popleft())
                                cnt = cnt + 1
                                if cnt >= self.batch_size/4 and len(self.data_buffer) > self.batch_size:
                                    t2 = time.time()
                                    print("batch i:{},collecting finished,samples:{},time_used:{:.3f}".format(
                                        i + 1, cnt, t2 - t1))
                                    break
                            except:
                                break
                        if cnt >= self.batch_size/4 and len(self.data_buffer) > self.batch_size:
                            break
                        else:
                            t2 = time.time()
                            print("batch i:{},collecting continue,samples:{},time_used:{:.3f}".format(
                                i + 1, cnt, t2 - t1))
                        time.sleep(2)
                else:
                    while True:
                        while self.data_queue.empty():
                            time.sleep(1)
                        item = self.data_queue.get()
                        self.data_buffer.append(item)
                        cnt = cnt + 1
                        if cnt > self.batch_size and len(self.data_buffer) > self.batch_size:
                            break
                t2 = time.time()
                print("batch i:{}, data_buffer_size:{},time_used:{:.3f}".format(i + 1, len(self.data_buffer), t2 - t1))
                for i  in range(5):
                    try:
                        loss, entropy = self.policy_update()
                        break
                    except:
                        continue
                state = {'state_dict': self.policy_value_net.policy_value_model.state_dict(),
                         'optim_dict': self.policy_value_net.optimizer.state_dict(),
                         'loss': loss,
                         'entropy': entropy}
                torch.save(state, self.model_file + '.undone')
                shutil.move(self.model_file + '.undone', self.model_file)
                # upload(data_server_url, self.model_file)
                torch.save(state, self.model_file + '.undone')
                shutil.copy(self.model_file + '.undone', os.path.join(TunnelPath, os.path.split(self.model_file)[-1]))
        except KeyboardInterrupt:
            print('\n\rquit')

    def release(self):
        for proc in self.collect_procs:
            proc.terminate()
            proc.join()
        for proc in self.eval_procs:
            proc.terminate()
            proc.join()


# Training settings
def parse_arguments():
    parser = argparse.ArgumentParser(description='AlphaZeroGomoku')
    parser.add_argument('--is_dist', metavar='DIST', default='1',
                        choices=['1', '0'],
                        help='run mode: dist or local')
    parser.add_argument('--role', metavar='ROLE', default='master',
                        choices=['worker', 'master', 'evaluator'],
                        help='run role: worker or master')
    parser.add_argument('--data_server_url', metavar='URL', default=DIST_DATA_URL,
                        type=str,
                        help='data_server_url')
    parser.add_argument('--is_resume', metavar='RESUME', default='0',
                        choices=['1', '0'],
                        help='run mode: dist or local')
    args = parser.parse_args()
    return args


def main(args):
    if args.is_resume == '0':
        if os.path.exists(args.model_file):
            os.remove(args.model_file)
    if args.is_dist == '0':
        training_pipeline = TrainPipeline()
        print('start collecting')
        training_pipeline.collect_selfplay_data(is_distributed=False)
        print('start evaluating')
        training_pipeline.policy_evaluate()
        training_pipeline.init_model()
        print('start training')
        training_pipeline.train(is_distributed=False)
        training_pipeline.release()
    elif args.is_dist == '1':
        if args.role == 'master':
            '''
            serv = threading.Thread(target=start_server, args=())
            serv.setDaemon(True)
            serv.start()
            '''
            training_pipeline = TrainPipeline()
            training_pipeline.init_model()
            print('start dist training')
            training_pipeline.train(is_distributed=True, data_server_url=args.data_server_url)
        elif args.role == 'worker':
            training_pipeline = TrainPipeline()
            print('start dist collecting')
            training_pipeline.collect_selfplay_data(is_distributed=True, data_server_url=args.data_server_url)
        elif args.role == 'evaluator':
            training_pipeline = TrainPipeline()
            print('start dist evaluating')
            training_pipeline.policy_evaluate()
            training_pipeline.eval(is_distributed=True, data_server_url=args.data_server_url)


if __name__ == '__main__':
    main(parse_arguments())
