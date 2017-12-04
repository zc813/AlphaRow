from alphazero.models import new_model
from player import AIPlayer
from queue import Empty as QueueEmptyError
import numpy as np

class GamePlayer(object):
    def __init__(self, game_class, player_num, policy_width, *args, **kwargs):
        self.game_class = game_class
        self.game_args = args
        self.game_kwargs = kwargs
        self.policy_width = policy_width
        self.player_num = player_num

    def new_game(self, players=None):
        game = self.game_class(*self.game_args, **self.game_kwargs)
        player_num = game.get_player_num()
        if player_num != len(players) or player_num != self.player_num:
            raise ValueError("Players input do not match game player number!")
        for i, player in enumerate(players):
            game.set_player(i, player)

        return game

    def evaluate(self, best_logic, new_logic, game_rounds):
        players = [AIPlayer(0, new_logic, monitor=False)]
        players += [AIPlayer(i, best_logic, monitor=False) for i in range(1, self.player_num)]
        game = self.new_game(players=players)
        # cumulative_result = [0] * len(self.player_num)
        cumulative_result = np.zeros((self.player_num, ), dtype=float)
        for i in range(game_rounds):
            result = game.start()
            cumulative_result += np.array(result)
            # for i, single_result in enumerate(result):
            #     cumulative_result[i] += result

        avg_result = cumulative_result / game_rounds
        # avg_result = [result * 1.0 / game_rounds for result in cumulative_result]
        return avg_result

    def self_play(self, logic, game_rounds):
        players = [AIPlayer(i, logic, monitor=True) for i in range(self.player_num)]
        game = self.new_game(players=players)
        statuses = list()
        actions = list()
        results = list()
        for i in range(game_rounds):
            result = game.start()
            history = players[0].get_singleton_history()
            for player_idx, status, action in history:
                statuses.append(status)
                actions.append(action) # dict
                results.append(result[player_idx])
            print("PLAY | Round %d/%d. %d entries of history." % (i, game_rounds, len(history)))
            players[0].reset_history()
        x = np.array(statuses,)
        y = np.zeros((len(actions), self.policy_width+1,), float)
        for i, action in enumerate(actions):
            for key, value in action.items():
                y[i, key] = value
            y[i, -1] = results[i]
        return x, y

def max_eval_fn(results):
    if max(range(len(results)), key=lambda i:results[i]) == 0:
        return True
    return False
