from alphazero.models import new_model
from player import AIPlayer
from queue import Empty as QueueEmptyError
import numpy as np

class GamePlayer(object):
    def __init__(self, game_class, player_num, input_shape, policy_width, *args, **kwargs):
        self.game_class = game_class
        self.game_args = args
        self.game_kwargs = kwargs
        self.input_shape = input_shape
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
            print("Round %d/%d. %d entries of history." % (i, game_rounds, len(history)))
            players[0].reset_history()
        x = np.array(statuses,)
        y = np.zeros((len(actions), self.policy_width+1,), float)
        for i, action in enumerate(actions):
            for key, value in action.items():
                y[i, key] = value
            y[i, -1] = results[i]
        return x, y


    def self_play_worker(self,game):
        result = game.start()
        return game.history, result


def worker(game_player, best_logic, latest_logic, data_queue, param_queue, evaluation_fn=None):
    evaluation_fn = evaluation_fn or max_eval_fn

    while True:
        params = None
        while True:
            try:
                params = param_queue.get(block=False)
            except QueueEmptyError:
                break

        if params is not None:
            # print("Evaluation_worker shape: %s, length: %d" % (str(getattr(params, 'shape')), len(params)))
            latest_logic.model.set_weights(params)
            # results = game_player.evaluate(best_logic, latest_logic, 10)
            # if evaluation_fn(results):
            if True:
                best_logic.model.set_weights(latest_logic.model.get_weights())

        print("Starting self_playing...")
        data = game_player.self_play(best_logic, 4)
        print("Self-playing ended. Sending data,", data[0].shape, data[1].shape)
        data_queue.put(data)

def max_eval_fn(results):
    if max(range(len(results)), key=results) == 0:
        return True
    return False