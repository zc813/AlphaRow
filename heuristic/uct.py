from heuristic.heuristic import Heuristic
import math

class UCT(Heuristic):
    def __init__(self, c=1.0):
        super(UCT, self).__init__()
        self.append_default_values(ucb1_1=math.inf, ucb1_2=math.inf)
        self.c = c

    def root_value(self):
        return dict(visited=1, ucb1_1=-math.inf,ucb1_2=-math.inf)

    def update_appended_value(self, values:dict, scores:list, *args, **kwargs):
        parent_values = kwargs['parent_values']
        parent_visited = parent_values['visited'] if parent_values is not None else self.total_visits
        values['ucb1_1'] = self.ucb1(avg=values['score1']/(values['score1']+values['score2']),
                                     total_visit=parent_visited, node_visit=values['visited'])
        values['ucb1_2'] = self.ucb1(avg=values['score2']/(values['score1']+values['score2']),
                                     total_visit=parent_visited, node_visit=values['visited'])

    def ucb1(self, avg=None, total_visit=None, node_visit=None):
        if node_visit == 0:
            return math.inf
        if total_visit == None:
            total_visit = self.total_visits
        return avg + self.c * math.sqrt(math.log(total_visit) / node_visit)

    def cmp(self, a, b, player):
        if a is None:
            return True
        if b is None:
            return False

        if player == 1:
            cmp_a = a['ucb1_1']
            cmp_b = b['ucb1_1']
        elif player == 2:
            cmp_a = a['ucb1_2']
            cmp_b = b['ucb1_2']
        else:
            raise ValueError('`player` has to be 1 or 2')

        if cmp_a >= cmp_b:
            return False
        return True

    def cmp_result(self, a, b, player):
        if a is None:
            return True
        if b is None:
            return False

        cmp_a = a['visited']
        cmp_b = b['visited']

        if cmp_a >= cmp_b:
            return False
        return True