from copy import deepcopy

class Status(object):
    def get_current_player_idx(self):
        raise NotImplementedError

    def get_available_actions(self):
        raise NotImplementedError

    def get_round(self):
        raise NotImplementedError

    def perform(self, action):
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError

    def get_result_score(self):
        raise NotImplementedError

    def to_number(self):
        raise NotImplementedError

    def copy(self):
        return deepcopy(self)

class Player(object):
    def get_action(self, status):
        raise NotImplementedError

class Game(object):
    def get_player_num(self):
        # 最好支持多玩家，比如斗地主
        raise NotImplementedError

    def set_player(self, idx, player):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    # def add_callback(self, callback):
    #     if hasattr(self, 'callbacks'):
    #         self.callbacks.append(callback)
    #     else:
    #         self.callbacks = [callback]
    #
    # def on_game_begin(self):
    #     for callback in self.callbacks:
    #         callback.on_game_begin(self.status)
    #
    # def on_game_end(self):
    #     for callback in self.callbacks:
    #         callback.on_game_end(self.status)
    #
    # def on_round_begin(self):
    #     self._current_player_idx = self.status.get_current_player_idx()
    #     for callback in self.callbacks:
    #         callback.on_round_begin(self._current_player_idx, self.status)
    #
    # def on_round_end(self):
    #     for callback in self.callbacks:
    #         callback.on_round_begin(self._current_player_idx, self.status)

# class GameCallback(object):
#     def on_game_begin(self, status):
#         pass
#
#     def on_game_end(self, status):
#         pass
#
#     def on_round_begin(self, player_idx, status):
#         pass
#
#     def on_round_end(self, player_idx, status):
#         pass