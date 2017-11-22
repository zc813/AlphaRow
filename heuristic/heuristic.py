import copy

class Heuristic(object):
    def __init__(self, num_players):
        self.num_players = num_players
        self.node_default_values = dict(visited=0, scores=[0]*self.num_players)
        self.total_visits = 0

    def append_default_values(self, **kwargs):
        for key, value in kwargs.items():
            self.node_default_values[key] = value

    def init_value(self) -> dict:
        return copy.deepcopy(self.node_default_values)

    def root_value(self):
        return dict(visited=1)

    def update_total_visits(self):
        self.total_visits += 1

    def update_value(self, values:dict, scores:list, **kwargs):
        # assert scores[0] + scores[1] <= 1
        values['visited'] += 1
        for i in range(self.num_players):
            values['scores'][i] += scores[i]
        self.update_appended_value(values, scores, **kwargs)

    def update_appended_value(self, values: dict, scores: list, **kwargs):
        pass

    def cmp(self, a, b, player_idx):
        """
        Used in selection. Usually compares UCT values or equivalents.
        :return: `True` if the value of `a` is less than `b`
        """
        raise NotImplementedError

    def cmp_result(self, a, b, player_idx):
        """
        Used in action choosing after simulations. Usually compares visited times.
        :return: `True` if the value of `a` is less than `b`
        """
        raise NotImplementedError

    def get_result(self, values, player_idx):
        raise NotImplementedError