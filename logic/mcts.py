from logic.AI_logic import AI_Logic
from logic.treenode import TreeNode
from heuristic.uct import UCT
from random import choice

class MCTS_logic(AI_Logic):
    def __init__(self, heuristics=None):
        self.heuristics = heuristics or UCT()
        self.rollout_policy = 'random'

    def get_move(self, board, player, iterations = 3000):
        tree = self._run(board, iterations, player)

        # debug code
        availables = [child.element for child in tree.children]
        for h in range(board.height-1, -1, -1):
            buffer = ''
            for w in range(board.width):
                i = board.location_to_move([h,w])
                out = 'VISIT SCORE UCB_1'
                if i in availables:
                    loc = availables.index(i)
                    values = tree.children[loc].values
                    # out = '%.3f' % values['ucb1_2' if player == 2 else 'ucb1_1']
                    out = '%5d' % values['visited']
                    out += ' %.3f' % (values['score%d'%player] / (values['score1'] + values['score2']))
                    out += ' %.3f' % values['ucb1_%d'%player]
                buffer += '%d,%d:%s\t' % (h, w, out)
            print(buffer)
        # end debug code

        best_node = self._select_best(tree, player)
        result = best_node.element

        if result is not None:
            return result
        else:
            raise RuntimeError

    def _run(self, board, iterations, player):
        rootnode = self._new_node(None)
        rootnode.values.update(self.heuristics.root_value())
        for i in range(iterations):
            # selection & expansion
            simulation = board.copy()
            leafnode = self._selection(simulation, rootnode, player)

            # simulation
            winner = self._rollout(simulation)

            # back propagation
            self._back_propagate(leafnode, self.heuristics.roundscore(winner))

        return rootnode

    def _selection(self, board: 'Board', root: 'TreeNode', player):
        if root.is_leaf():
            if root.values['visited'] == 0:
                return root
            else:
                # expand when all children visited
                self._expand(board, root)
                # no available expansion
                if root.is_leaf():
                    return root
        if self._isterminal(board):
            return root

        best = None
        # player = self._reverse_player(player)
        for child in root.children:
            if self.heuristics.cmp(getattr(best, 'values', None), child.values, player):
                best = child

        board.do_move(best.element)
        return self._selection(board, best, self._reverse_player(player))

    def _isterminal(self, board : 'Board'):
        if board.game_end()[0]:
            return True
        return False

    def _rollout(self, board : 'Board') -> int:
        if self.rollout_policy == 'random':
            end, winner = board.game_end()
            while board.availables and not end:
                board.do_move(choice(board.availables))
                end, winner = board.game_end()
            return winner
        else:
            raise NotImplementedError

    def _back_propagate(self, leafnode, scores=[0,0]):
        self.heuristics.update_total_visits()
        l = list()
        while leafnode is not None:
            l.append(leafnode)
            leafnode = leafnode.parent
        while l:
            leafnode = l.pop()
            self.heuristics.update_value(leafnode.values, scores, parent_values=leafnode.parent.values if leafnode.parent is not None else None)

    def _expand(self, board, node):
        for move in board.availables:
            node.add_child(self._new_node(move))

    def _new_node(self, element):
        new_node = TreeNode(element)
        new_node.values.update(self.heuristics.init_value())
        return new_node

    def _select_best(self, rootnode, player) -> 'TreeNode':
        best = rootnode.children[0]
        for child in rootnode.children:
            if self.heuristics.cmp_result(best.values, child.values, player):
                best = child
        return best