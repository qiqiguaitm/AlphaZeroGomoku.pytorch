import torch
from game import Board, Game
from policy_value_net import PolicyValueNet
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphazero import MCTSPlayer
from negamax import NegamaxPlayer
import viewer
import numpy as np
import sys

class Gomoku(object):
    def __init__(self):
        self.width, self.height = 19, 19
        feature_planes = 8
        n = 5
        model_file = 'checkpoint_best.pth.tar'
        self.board = Board(width=self.width, height=self.height, n_in_row=n)
        self.board.init_board(start_player=0)
        # checkpoint = torch.load(model_file)
        # best_policy_model = PolicyValueNet(width, height, feature_planes, mode='eval', checkpoint=checkpoint)
        # ai_player = MCTSPlayer(best_policy_model.policy_value_fn, c_puct=5,
        #                         n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (its much weaker even with a larger n_playout)
        # ai_player = MCTS_Pure(c_puct=5, n_playout=1000)
        ai_player = NegamaxPlayer(cmd_path='negamax/build/renju')
        self.ai_player = ai_player
        self.waiting_for_play = True
        self.chessboard = np.zeros((self.height,self.width))

    def play(self,chesspos):
        if not type(chesspos) is int:
            move = self.board.location_to_move(chesspos)
        else:
            move = chesspos
        self.board.do_move(move)
        self.waiting_for_play = False

    def get_chessboard(self):
        for i in range(self.height):
                for j in range(self.width):
                    loc = i * self.width + j
                    p = self.board.states.get(loc, 0)
                    self.chessboard[i,j] = p
        return self.chessboard

    def run(self,ai_fist=False):
        if ai_fist:
            ai_role = 1
            human_role = 2
            self.ai_player.set_player_ind(ai_role)
            chesspos = self.ai_player.get_action(self.board)
            self.play(chesspos)
            self.waiting_for_play = True
        else:
            human_role = 1
            ai_role = 2
            self.ai_player.set_player_ind(ai_role)
        self.ui = viewer.UI(pressaction=self.play, chessboardinfo=self.get_chessboard(), human_role=human_role)
        self.ui.start()
        end, winner = self.board.game_end()
        while not end:
            while self.waiting_for_play:
                pass
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    self.end_game(role='Human')
                else:
                    self.end_game(role='Nobody')
                break

            chesspos =  self.ai_player.get_action(self.board)
            self.play(chesspos)

            end, winner = self.board.game_end()
            self.ui.setchessboard(self.get_chessboard())
            if end:
                if winner != -1:
                    self.end_game(role='AIPlayer')
                else:
                    self.end_game(role='Nobody')
                break
            self.waiting_for_play = True
        
    def end_game(self, role):
        if role == 'Nobody':
            print("Tie")
        else:
            print(role + " Win")
        self.ui.gameend(role)


if __name__=='__main__':
    gomoku = Gomoku()
    print('game start ......')
    gomoku.run(ai_fist=False)
    print('game over.')
