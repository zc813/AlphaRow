from logic.mcts import MCTSLogic
from heuristic.puct_variant import PUCT
import numpy as np

class ModelBasedMCTSLogic(MCTSLogic):
    def __init__(self, model, heuristics=None):
        super(ModelBasedMCTSLogic, self).__init__(heuristics or PUCT(), self.evaluation_policy)
        self.model = model

    def evaluation_policy(self, status, node):
        prediction = self.model.predict(np.expand_dims(status.to_number(),0))
        policy = prediction[0,:-1]
        value = prediction[0,-1]
        for child in node.children:
            child.values['prior_probability'] = policy[child.element]
            self.heuristics.update_appended_value(child.values, None, node.values)
        scores = [-value] * self.heuristics.num_players
        scores[1-status.get_current_player_idx()] = value   # does not support multiple players
        return scores

    def _selection(self, status, root, player_idx):
        if root.is_leaf():
            self._expand(status, root)
            return root
        if self._isterminal(status):
            return root

        best = None
        for child in root.children:
            if self.heuristics.cmp(getattr(best, 'values', None), child.values, player_idx):
                best = child

        status.perform(best.element)
        return self._selection(status, best, status.get_current_player_idx())