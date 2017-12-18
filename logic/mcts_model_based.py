from logic.mcts import MCTSLogic
from heuristic.puct_variant import PUCT
import numpy as np
import math

class ModelBasedMCTSLogic(MCTSLogic):
    def __init__(self, model, heuristics=None, iterations=1600, explore_rounds=-1, verbose=False):
        super(ModelBasedMCTSLogic, self).__init__(heuristics or PUCT(), self.evaluation_policy, iterations=iterations)
        self.model = model
        self.exploring = explore_rounds
        self.verbose = verbose

    def evaluation_policy(self, status, node):
        prediction = self.model.predict(np.expand_dims(status.to_number(),0))
        policy = prediction[0,:-1]
        value = prediction[0,-1]
        #数十局对战mcts胜率55%，超过之前数十万局的水平（45%）；1000局达到85%。
        end, winner = status.game_end()
        if end:
            if winner != -1:
                value = 1 if status.get_current_player_idx() == winner else -1
            else:
                value = 0
        if self.iteration == 0 and (self.exploring < 0 or status.get_round() < self.exploring):
            policy *= 0.75
            policy += np.random.dirichlet(np.ones(len(policy))*0.3) * 0.25
        for child in node.children:
            child.values['prior_probability'] = policy[child.element]
            self.heuristics.update_appended_value(child.values, None, node.values)
        scores = [-value] * self.heuristics.num_players
        # scores[1-status.get_current_player_idx()] = value   # does not support multiple players
        scores[status.get_current_player_idx()] = value  # i think this is the right one
        return scores

    def _selection(self, status, root, player_idx):
        if root.is_leaf():
            self._expand(status, root)
            return root
        if self._isterminal(status):
            return root

        best = None
        # for child in root.children:
        #     if self.heuristics.cmp(getattr(best, 'values', None), child.values, player_idx):
        #         best = child
        # 原来是每次比较都要重新计算两个值，
        # 改为在 selection 里获取值并且保存，这部分比较总时间降低了42.0%，
        # selection 时间降低 29.9%，MCTS 总时间降低 19.5%
        best_value = -math.inf
        for child in root.children:
            current_value = self.heuristics.get_value(child.values, player_idx)
            if current_value > best_value:
                best = child
                best_value = current_value

        status.perform(best.element)
        return self._selection(status, best, status.get_current_player_idx())