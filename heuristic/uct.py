from heuristic.heuristic import Heuristic
import math

class UCT(Heuristic):
    def __init__(self, num_player=2, c=1.4):
        super(UCT, self).__init__(num_player)
        self.append_default_values(
            # ucb1_1=math.inf,
            # ucb1_2=math.inf,
                                   parent_values=None)
        self.c = c

    def root_value(self):
        return dict(visited=1, ucb1_1=-math.inf,ucb1_2=-math.inf)

    def update_appended_value(self, values:dict, scores:list, parent_values=None, **kwargs):
        values['parent_values']=parent_values
        # if parent_values is None:
        #     return
        # parent_visited = parent_values['visited'] #if parent_values is not None else self.total_visits  # this might introduce bugs
        # values['ucb1_1'] = self.ucb1(avg=1.0*values['scores'][0]/values['visited'],
        #                              total_visit=parent_visited, node_visit=values['visited'])
        # values['ucb1_2'] = self.ucb1(avg=1.0*values['scores'][1]/values['visited'],
        #                              total_visit=parent_visited, node_visit=values['visited'])

    def ucb1(self, avg=None, total_visit=None, node_visit=None):
        if node_visit == 0:
            return math.inf
        if total_visit == None:
            total_visit = self.total_visits
        return avg + self.c * math.sqrt(math.log(total_visit) / node_visit)

    def cmp(self, a, b, player_idx):
        if a is None:
            return True
        if b is None:
            return False
        value_a = self.ucb1(
            avg=1.0 * a['scores'][player_idx] / a['visited'] if a['visited'] != 0 else 0,
            total_visit=getattr(a['parent_values'], 'visited', None),
            node_visit=a['visited'], )
        value_b = self.ucb1(
            avg=1.0 * b['scores'][player_idx] / b['visited'] if b['visited'] != 0 else 0,
            total_visit=getattr(b['parent_values'], 'visited', None),
            node_visit=b['visited'], )
        if value_a >= value_b:
            return False
        return True

    def get_value(self, a, player_idx):
        if a is None:
            return -math.inf
        return self.ucb1(
            avg=1.0 * a['scores'][player_idx] / a['visited'] if a['visited'] != 0 else 0,
            total_visit=getattr(a['parent_values'], 'visited', None),
            node_visit=a['visited'], )

    def cmp_result(self, a, b, player_idx=None):
        if a is None:
            return True
        if b is None:
            return False

        cmp_a = a['visited']
        cmp_b = b['visited']

        if cmp_a >= cmp_b:
            return False
        return True

    def get_result(self, values, player_idx=None):
        return values['visited']