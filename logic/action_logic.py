from random import choice

class ActionLogic(object):
    def get_action(self, status, player_idx):
        raise NotImplementedError('`get_move` must be defined!')

    def set_player_num(self, num):
        self.player_num = num

    def get_player_num(self):
        return getattr(self, 'player_num', 2)

class Random_logic(ActionLogic):
    def get_action(self, status, player_idx):
        return choice(status.availables)