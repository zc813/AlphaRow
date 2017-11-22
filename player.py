from interface import Player

class AIPlayer(Player):
    def __init__(self, player_idx, logic, monitor=False):
        self.logic = logic
        self.monitor = monitor
        self.player_idx = player_idx
        if monitor:
            self.history = list()

    def get_action(self, status,):
        best_action = self.logic.get_action(status, self.player_idx)

        if self.monitor:
            action_results = self.logic.get_children_results() # dict
            self.history.append((status.to_number(), action_results))    # (s, pi, z)

        return best_action

    def get_history(self,):
        return self.history