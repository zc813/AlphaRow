from logic.action_logic import ActionLogic
import numpy as np
from random import choice

class Q_Learning(ActionLogic):
    def __init__(self, space_width=9, action_num=9, player_num=2):
        self.action_num = action_num
        self.set_player_num(player_num)
        states_num = (player_num + 1) ** space_width
        self._qtable = np.zeros((states_num, action_num, player_num))
        self._states_weights = np.ndarray((space_width, ), dtype=np.int64)
        for i in range(space_width):
            self._states_weights[i] = (player_num + 1) ** i

    def get_action(self, status, player_idx):
        state = self._get_state(status, player_idx)
        print(state, *self._qtable[state])
        return self._choose_best_action(status, player_idx)

    def train(self, board, iterations,):
        for i in range(iterations):
            self._play(board.copy())

    def _play(self, simulation,epsilon=0.9,  # 1.0 for solely greedy policies
              alpha=0.1,
              gamma=0.9):
        player = simulation.get_current_player_idx()+1
        state = self._get_state(simulation, player)
        if np.random.random() <= epsilon:
            move = self._choose_best_action(simulation, player)
        else:
            move = choice(simulation.availables)
        simulation.perform(move)
        new_state = self._get_state(simulation, player)
        end, winner = simulation.game_end()
        i = player-1
        r = self._get_reward(winner=winner, end=end, player=i+1)
        q_current_state = self._qtable[state,:, i]
        q_new_state = self._qtable[new_state,:,i]
        self._qtable[state, self._move_to_action(move), i] += alpha * (
        r + gamma * np.max(q_new_state) - q_current_state[self._move_to_action(move)])

        # propagate the opponent's state
        i = 1-i
        self._qtable[state, :, i] = self._qtable[new_state, :, i]    # this performs better, which does not make sense
        # self._qtable[state, :, i] = np.maximum(self._qtable[state, :, 1], self._qtable[new_state, :, i])
        if not end:
            return self._play(simulation)

    def _get_reward(self, winner, player, end=None):
        if end:
            if winner == player:
                return 1.0 # wins
            elif winner > 0:
                return 0.0 # loses
            else:
                return 0.5 # tie
        else:
            return 0.0     # not end

    def _choose_best_action(self, board, player):
        actions = self._qtable[self._get_state(board, player),:,player-1]
        availables = np.zeros(self.action_num)
        availables[board.availables] = 1
        action_idx = np.argmax(actions + availables - 1)
        return self._action_to_move(action_idx)

    def _get_state(self, board, player):
        state = self._states_weights.copy()
        for key, value in board.states.items():
            state[key] *= 0 if value==1 else 2  # mute this in Naive Game
        return np.sum(state)

    def _get_channel(self, player):
        return 0

    def _action_to_move(self, action_idx):
        return action_idx

    def _move_to_action(self, move):
        return move