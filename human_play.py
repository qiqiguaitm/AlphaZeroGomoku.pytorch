# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import torch
from game import Board, Game
from policy_value_net import PolicyValueNet
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphazero import MCTSPlayer
from negamax import NegamaxPlayer


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]  # for python3
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 9, 9
    feature_planes = 8
    model_file = 'checkpoint_best.pth.tar'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        checkpoint = torch.load(model_file)
        # best_policy_model = PolicyValueNet(width, height, feature_planes, mode='eval', checkpoint=checkpoint)
        # ai_player = MCTSPlayer(best_policy_model.policy_value_fn, c_puct=5,
        #                         n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (its much weaker even with a larger n_playout)
        # ai_player = MCTS_Pure(c_puct=5, n_playout=1000)
        ai_player = NegamaxPlayer(cmd_path='negamax/build/renju')
        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, ai_player, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
