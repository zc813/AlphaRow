# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:13:35 2017
@author: sjxn2423

任务： 实现MCTS_player这个类
"""

from __future__ import print_function
from logic.mcts import MCTSLogic
from logic.mcts_model_based import ModelBasedMCTSLogic
from logic.qlearning import Q_Learning
from alphazero.models import new_model
import interface
import numpy as np
import copy
from player import AIPlayer

class Board(interface.Status):
    """
    board for game
    """
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {} # board states, key:move, value: player as piece type
        self.n_in_row = int(kwargs.get('n_in_row', 5)) # need how many pieces in a row to win
        self.players = [1, 2] # player1 and player2
        self.current_player = self.players[1]
        self.winner=-1

    def get_available_actions(self):
        return self.availables

    def is_terminal(self) -> bool:
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if(len(moved) < self.n_in_row + 2):
            self.winner = -1
            return False

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                self.winner = player
                return True

            if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                self.winner = player
                return True

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                self.winner = player
                return True

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                self.winner = player
                return True
        self.winner = -1
        return False

    def get_result_score(self):
        raise NotImplementedError

    def to_number(self):
        array = np.zeros((self.height, self.width, 2), dtype=np.uint8)
        for key, player in self.states.items():
            h, w = self.move_to_location(key)
            array[h,w,player-1] = 1
        return array

    def init_board(self):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row)
        self.availables = list(range(self.width * self.height)) # available moves
        self.states = {} # key:move as location on the board, value:player as pieces type
        self.current_player = self.players[1]
        self.winner = -1

    def move_to_location(self, move):
        """
        一维编号move和二维坐标location的转化
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move  // self.width
        w = move  %  self.width
        return [h, w]

    def location_to_move(self, location):
        if(len(location) != 2):
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if(move not in range(self.width * self.height)):
            return -1
        return move

    def perform(self, action):
        self.states[action] = self.current_player
        self.availables.remove(action)
        #执行move之后，切换current player
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]

    def has_a_winner(self):
        return self.is_terminal(), self.winner

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            # print("Game end. Tie")
            return True, -1
        return False, -1

    def get_current_player_idx(self):
        return self.current_player-1

    def copy(self) -> 'Board':
        return copy.deepcopy(self)


class MCTS_Player(interface.Player):
    """TODO: 实现这个MCTS player"""
    def __init__(self, player, logic=None):
        self.player = player
        if logic is None:
            raise ValueError('`logic` must be specified.')
        self.logic = logic

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.logic.get_action(board, self.player-1)
            location = board.move_to_location(move)
            print("AI move: %d,%d\n" % (location[0], location[1]))
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS"

class Human(interface.Player):
    """
    human player
    """

    def __init__(self, player):
        self.player = player

    def get_action(self, board):
        try:
            location = eval(input("Your move: ").replace('，',','))
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human"


class Game(interface.Game):
    """
    game server
    """
    def __init__(self, board, **kwargs):
        self.board = board
        self.n_in_row = int(kwargs.get('n_in_row', 5))

    def get_player_num(self):
        return 2

    def set_player(self, idx, player):
        pass

    def start(self):
        self.board.init_board()
        p1, p2 = self.board.players

        ai = MCTS_Player(p2, MCTSLogic())
        model = new_model((self.board.height, self.board.width, 2), self.board.height*self.board.width)
        smarterai = AIPlayer(p2-1, ModelBasedMCTSLogic(model), True)
        # human = MCTS_Player(p2, MCTS_logic())
        human = Human(p1)
        players = {}
        players[p2] = smarterai
        players[p1] = human
        # ai.logic.train(self.board, 20000)
        while(1):
            self.board.init_board()
            self.graphic(self.board, human, smarterai)
            while(1):
                current_player = self.board.get_current_player_idx()+1
                # print(ai.logic._qtable[ai.logic._get_state(self.board,current_player)])
                player_in_turn = players[current_player]
                move = player_in_turn.get_action(self.board)
                self.board.perform(move)
                self.graphic(self.board, human, ai)
                end, winner = self.board.game_end()
                print(smarterai.get_history())
                if end:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                    break

    def graphic(self, board, human, ai):
        """
        Draw the board and show game info
        """
        width = board.width
        height = board.height

        print("Human Player", getattr(human, 'player', 0), "with X".rjust(3))
        print("AI    Player", getattr(ai, 'player', 0), "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == getattr(human, 'player', 0):
                    print('X'.center(8), end='')
                elif p == getattr(ai, 'player', 0):
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')


def run():
    # 可以先在 width=3, height=3, n=3 这种最简单的case下开发实验
    # 这种case下OK之后，再测试下 width=6, height=6, n=4 这种情况
    n = 4
    try:
        board = Board(width=6, height=6, n_in_row=n)
        game = Game(board, n_in_row=n)
        game.start()
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
