import subprocess
import json
class NegamaxPlayer(object):
    """AI player based on MCTS"""

    def __init__(self,cmd_path,search_depth=2,time_limit=5500,threads=1):
        self.cmd_path = cmd_path
        self.search_depth = search_depth
        self.time_limit = time_limit
        self.threads = threads

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        return None

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) == board.width * board.height:
            return (board.width * board.height)/2
        elif len(sensible_moves) > 0:
            state_line = board.state_line()
            cmd = self.cmd_path + ' -b ' + str(board.height) + ' -s ' + state_line \
                  + ' -p ' + str(self.player) + ' -d ' + str(self.search_depth)
            sub = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            sub.wait()
            ret = sub.stdout.read()
            ret = json.loads(ret)
            move_c = ret['result']['move_c']
            move_r = ret['result']['move_r']
            move = int(move_r) * board.width + int(move_c)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Negamax {}".format(self.player)