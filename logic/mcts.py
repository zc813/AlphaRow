from logic.action_logic import ActionLogic
from logic.treenode import TreeNode
from heuristic.uct import UCT
from random import choice
from interface import Status

class MCTSLogic(ActionLogic):
    def __init__(self, heuristics=None, evaluation_policy=None, iterations=1600):
        self.heuristics = heuristics or UCT()
        self.evaluation_policy = evaluation_policy or self._rollout_evaluation
        self.prev_tree = None
        self.iterations = iterations
        self.iteration = 0
        self.verbose = False

    def get_action(self, status:Status, player_idx, reuse=True):
        """
        TODO: Supports reuse of tree. Note that the new root might be one of the children or grand children of the current root.
        :param status:
        :param player_idx:
        :param iterations:
        :param reuse:
        :return:
        """
        tree = self._run(status, self.iterations, player_idx, rootnode=None)

        # debug code
        if self.verbose:
            availables = [child.element for child in tree.children]
            for h in range(status.height-1, -1, -1):
                buffer = ''
                for w in range(status.width):
                    i = status.location_to_move([h, w])
                    out = 'VISIT SCORE UCB_1'
                    if i in availables:
                        loc = availables.index(i)
                        values = tree.children[loc].values
                        # out = '%.3f' % values['ucb1_2' if player == 2 else 'ucb1_1']
                        out = '%5d' % values['visited']
                        out += ' %.3f' % (values['scores'][player_idx])
                        # out += ' %.3f' % (values['uct'][player_idx])
                    buffer += '%d,%d:%s\t' % (h, w, out)
                print(buffer)
        # end debug code

        best_node = self._select_best(tree, player_idx)
        self.prev_tree = tree
        result = best_node.element

        if result is not None:
            return result
        else:
            raise RuntimeError

    def get_children_results(self):
        if self.prev_tree is None:
            raise ValueError("Tree not initialized!")
        result = dict()
        for child in self.prev_tree.children:
            result[child.element] = self.heuristics.get_result(child.values)
        return result

    def _run(self, status, iterations, player_idx, rootnode=None):
        if rootnode is None:
            rootnode = self._new_node(None)
            #TODO: investigate the visited number equal to 1
            rootnode.values.update(self.heuristics.root_value())
        for self.iteration in range(iterations):
            # selection & expansion
            simulation = status.copy()
            leafnode = self._selection(simulation, rootnode, player_idx)

            # simulation
            scores = self.evaluation_policy(simulation, leafnode)

            # back propagation
            self._back_propagate(leafnode, scores)

        return rootnode

    def _selection(self, status, root: 'TreeNode', player_idx):
        if root.is_leaf():
            if root.values['visited'] == 0:
                return root
            else:
                # expand when all children visited
                self._expand(status, root)
                # no available expansion
                if root.is_leaf():
                    return root
        if self._isterminal(status):
            return root

        best = None
        # for child in root.children:
        #     if self.heuristics.cmp(getattr(best, 'values', None), child.values, player_idx):
        #         best = child
        best_value = self.heuristics.get_value(None, 0)
        for child in root.children:
            current_value = self.heuristics.get_value(child.values, player_idx)
            if current_value > best_value:
                best = child
                best_value = current_value

        status.perform(best.element)
        return self._selection(status, best, status.get_current_player_idx())

    def _isterminal(self, status):
        if status.is_terminal():
            return True
        return False

    def _rollout_evaluation(self, status, node):
        winner = self._random_rollout(status)
        if winner  == -1:
            scores = [0] * self.heuristics.num_players
        else:
            scores = [-1] * self.heuristics.num_players
            scores[winner]=1
        return scores

    def _random_rollout(self, status) -> int:
        end, winner = status.game_end()
        while status.get_available_actions() and not end:
            status.perform(choice(status.get_available_actions()))
            end, winner = status.game_end()
        return winner

    def _back_propagate(self, leafnode, scores):
        self.heuristics.update_total_visits()
        l = list()
        while leafnode is not None:
            l.append(leafnode)
            leafnode = leafnode.parent
        while l:
            leafnode = l.pop()
            self.heuristics.update_value(leafnode.values, scores,
                                         parent_values=leafnode.parent.values
                                         if leafnode.parent is not None
                                         else None)

    def _expand(self, status, node):
        for move in status.get_available_actions():
            node.add_child(self._new_node(move))

    def _new_node(self, element):
        new_node = TreeNode(element)
        new_node.values.update(self.heuristics.init_value())
        return new_node

    def _select_best(self, rootnode, player_idx) -> 'TreeNode':
        best = rootnode.children[0]
        for child in rootnode.children:
            if self.heuristics.cmp_result(best.values, child.values, player_idx):
                best = child
        return best