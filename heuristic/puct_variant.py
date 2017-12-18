from heuristic.heuristic import Heuristic
import math

class PUCT(Heuristic):
    def __init__(self, num_player=2, c=5.0, ):
        super(PUCT, self).__init__(num_player)
        self.append_default_values(prior_probability=1.0,
                                   # puct=[math.inf]*self.num_players,
                                   parent_values=None)
        self.c = c

    def root_value(self):
        return dict(visited=1, puct=[-math.inf]*self.num_players)

    def update_appended_value(self, values:dict, scores:list, parent_values=None, **kwargs):
        # if parent_values is None:
        #     return
        # parent_visited = parent_values['visited']
        values['parent_values'] = parent_values
        # for i in range(self.num_players):
        #     values['puct'][i] = self.puct(avg=1.0*values['scores'][i]/values['visited'] if values['visited']!=0 else 0,
        #                                   probability=values['prior_probability'],
        #                                   total_visit=parent_visited,
        #                                   node_visit=values['visited'], )

    def puct(self, avg, probability, total_visit, node_visit):
        return avg + self.c * probability * math.sqrt(total_visit) / (1+node_visit)

    def cmp(self, a, b, player_idx):
        if a is None:
            return True
        if b is None:
            return False
        puct_a = self.puct(
            avg=1.0 * a['scores'][player_idx] / a['visited'] if a['visited'] != 0 else 0,
            probability=a['prior_probability'],
            total_visit=getattr(a['parent_values'], 'visited', 1),
            node_visit=a['visited'], )
        puct_b = self.puct(
            avg=1.0 * b['scores'][player_idx] / b['visited'] if b['visited'] != 0 else 0,
            probability=b['prior_probability'],
            total_visit=getattr(b['parent_values'], 'visited', 1),
            node_visit=b['visited'], )
        if puct_a >= puct_b:
            return False
        return True
        # return self._cmp(a, b, 'puct', player_idx)

    def get_value(self, a, player_idx):
        if a is None:
            return -math.inf
        return self.puct(
            avg=1.0 * a['scores'][player_idx] / a['visited'] if a['visited'] != 0 else 0,
            probability=a['prior_probability'],
            total_visit=getattr(a['parent_values'], 'visited', 1),
            node_visit=a['visited'], )

    def cmp_result(self, a, b, player_idx=None):
        """
        TODO: implement a exponentiated formula as described in METHODS.Play
        """
        return self._cmp(a, b, 'visited')

    def get_result(self, values, player_idx=None):
        """
        TODO: Fix not being updated when the child is not visited
        """
        return values['visited'] * 1.0 / values['parent_values']['visited']

    def _cmp(self, a, b, key, player_idx=None):
        if a is None:
            return True
        if b is None:
            return False
        cmp_a = self._get_values_by_key(a, key, player_idx)
        cmp_b = self._get_values_by_key(b, key, player_idx)
        if cmp_a >= cmp_b:
            return False
        return True

    def _get_values_by_key(self, values, key, player_idx=None):
        return values[key][player_idx] if player_idx is not None else values[key]
