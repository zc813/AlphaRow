from random import choice

class AI_Logic(object):
    def get_move(self, board, player):
        raise NotImplementedError('`get_move` must be defined!')

    def set_player_num(self, num):
        self._player_num = num

    def get_player_num(self):
        return getattr(self, 'player_num', 2)

    def _reverse_player(self, player):
        if player == 1:
            return 2
        elif player == 2:
            return 1
        return player

class Random_logic(AI_Logic):
    def get_move(self, board, player):
        return choice(board.availables)