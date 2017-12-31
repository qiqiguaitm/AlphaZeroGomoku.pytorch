import subprocess
import json
class NegamaxPlayer(object):
    """AI player based on MCTS"""

    def __init__(self,cmd_path,search_depth=-1,time_limit=5500,threads=1):
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
        if len(sensible_moves) == 19*19:
            return 180
        elif len(sensible_moves) > 0:
            state_line = board.state_line()
            cmd = self.cmd_path + ' -s ' + state_line + ' -p ' + str(self.player)
            sub = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            sub.wait()
            ret = json.loads(sub.stdout.read())
            move_c = ret['result']['move_c']
            move_r = ret['result']['move_r']
            move = int(move_r) * board.width + int(move_c)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Negamax {}".format(self.player)