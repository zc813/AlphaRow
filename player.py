from interface import Player

class AIPlayer(Player):
    singleton_history = list()

    def __init__(self, player_idx, logic, monitor=False):
        self.logic = logic
        self.monitor = monitor
        self.player_idx = player_idx
        if monitor:
            self.history = list()

    def get_action(self, status, ):
        best_action = self.logic.get_action(status, self.player_idx)

        if self.monitor:
            action_results = self.logic.get_children_results() # dict todo：俊潇把这个π取了对数然后softmax
            status_compressed = status.to_number()
            history_entry = (self.player_idx, status_compressed, action_results, status.get_round())
            self.history.append(history_entry)    # (s, pi, z)
            self.singleton_history.append(history_entry)

        return best_action

    def get_history(self,):
        return self.history

    def get_singleton_history(self, ):
        return self.singleton_history

    def reset_history(self, ):
        if self.monitor:
            self.history.clear()
        self.singleton_history.clear()