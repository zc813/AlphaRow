class Heuristic(object):
    def __init__(self):
        self.node_default_values = dict(visited=0, score1=0, score2=0)
        self.total_visits = 0

    def append_default_values(self, **kwargs):
        for key, value in kwargs.items():
            self.node_default_values[key] = value

    def init_value(self) -> dict:
        return self.node_default_values.copy()

    def update_total_visits(self):
        self.total_visits += 1

    def update_value(self, values:dict, scores:list, **kwargs):
        assert scores[0] + scores[1] <= 1
        values['visited'] += 1
        values['score1'] += scores[0]
        values['score2'] += scores[1]
        self.update_appended_value(values, scores, **kwargs)

    def root_value(self):
        return dict(visited=1)

    def roundscore(self, winner):
        if winner == -1:
            return [0.5, 0.5]
        elif winner == 1:
            return [1, -0.7]
        elif winner == 2:
            return [-0.7, 1]
        else:
            raise RuntimeError('Unexpected winner = ' + str(winner))

    def update_appended_value(self, values: dict, scores: list, **kwargs):
        pass

    def cmp(self, a, b, player):
        """
        :return: `True` if the value of `a` is less than `b`
        """
        raise NotImplementedError

    def cmp_result(self, a, b, player):
        return self.cmp(a, b, player)