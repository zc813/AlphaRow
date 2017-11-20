# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:13:35 2017
@author: sjxn2423

任务： 实现MCTS_player这个类
"""

from __future__ import print_function
from logic.mcts import MCTS_logic
from logic.qlearning import Q_Learning
import copy

class Board(object):
    """
    board for game
    """
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 4))
        self.height = int(kwargs.get('height', 4))
        self.max_rounds = int(kwargs.get('rounds', 50))
        self.pos = (0,0)
        self.availables = self._get_availables()
        self.states={0:1}
        self.player = 1
        self.rounds=0

    def do_move(self, move):
        if move not in self.availables:
            raise ValueError
        if move == 0:
            self.pos = (self.pos[0]-1, self.pos[1])
        elif move == 1:
            self.pos = (self.pos[0]+1, self.pos[1])
        elif move == 2:
            self.pos = (self.pos[0], self.pos[1]-1)
        elif move == 3:
            self.pos = (self.pos[0], self.pos[1]+1)
        elif move == 4:
            pass
        self.states = {self.pos[0]+self.pos[1]*self.width:1}
        self.rounds += 1
        self.availables = self._get_availables()
        self.player = 3 - self.player

    def _get_availables(self):
        availables = [4]
        if self.pos[0] > 0:
            availables.append(0)
        if self.pos[0] < self.width-1:
            availables.append(1)
        if self.pos[1] > 0:
            availables.append(2)
        if self.pos[1] < self.height-1:
            availables.append(3)
        return availables

    def has_a_winner(self):
        if self.pos[0] >= self.width - 1 and self.pos[1]>=self.height-1 and self.rounds<self.max_rounds:
            return True, 1
        elif self.rounds >= self.max_rounds:
            return True, 2
        else:
            return False, -1

    def game_end(self):
        return self.has_a_winner()

    def get_current_player(self):
        return self.player

    def copy(self) -> 'Board':
        return copy.deepcopy(self)


class MCTS_Player(object):
    """TODO: 实现这个MCTS player"""
    def __init__(self, player, logic=None):
        self.player = player
        if logic is None:
            raise ValueError('`logic` must be specified.')
        self.logic = logic

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.logic.get_move(board, self.player)
            print("AI move: %d\n" % (move))
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS"


class Human(object):
    """
    human player
    """

    def __init__(self, player):
        self.player = player

    def get_action(self, board):
        try:
            move = int(input("Your move: "))
            # move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human"

class Game(object):
    """
    game server
    """
    def __init__(self, board, **kwargs):
        self.board = board

    def start(self):
        ai = MCTS_Player(1, Q_Learning(space_width=self.board.width*self.board.height, action_num=5, player_num=2))
        # human = MCTS_Player(p2, MCTS_logic())
        human = Human(1)
        human1 = Human(2)
        players = [ai, human1]
        ai.logic.train(self.board, 1000)
        self.graphic(self.board)
        i = 0
        while(1):
            print('Player %d' % (i+1))
            move = players[i].get_action(self.board)
            i = 1 - i
            self.board.do_move(move)
            self.graphic(self.board)
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    print("Game end. Winner is %d" % winner)
                else:
                    print("Game end. Tie")
                break

    def graphic(self, board):
        """
        Draw the board and show game info
        """
        for j in range(board.height):
            out = ''
            for i in range(board.width):
                out += '-' if (i,j)!=board.pos else 'x'
            print(out)


def run():
    # 可以先在 width=3, height=3, n=3 这种最简单的case下开发实验
    # 这种case下OK之后，再测试下 width=6, height=6, n=4 这种情况
    try:
        board = Board()
        game = Game(board)
        game.start()
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
