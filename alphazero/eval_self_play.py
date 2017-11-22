from alphazero.models import new_model
from player import AIPlayer

class GamePlayer(object):
    def __init__(self, game_class, input_shape, policy_width, *args, **kwargs):
        self.game_class = game_class
        self.game_args = args
        self.game_kwargs = kwargs
        self.input_shape = input_shape
        self.policy_width = policy_width

    def new_game(self, parameters:'parameters or list of parameters'=None, players=None):
        game = self.game_class(self.game_args, self.game_kwargs)
        player_num = game.get_player_num()
        if parameters != None:
            players = list()
            if isinstance(parameters, list):
                if len(parameters) != player_num:
                    raise ValueError("The length of parameters (%d) does not equal to the number of players (%d)!" % (
                    len(parameters), player_num))
                for params in parameters:
                    players.append(self.new_player(params))
            else:
                for i in range(player_num):
                    players.append(self.new_player(parameters))

        if player_num != len(players):
            raise ValueError
        for i, player in enumerate(players):
            game.set_player(i, player)

        return game

    def new_player(self, parameters=None, monitor=False):
        model = new_model(self.input_shape, self.policy_width, parameters=parameters)
        return AIPlayer(monitor=monitor)

    def evaluate(self, list_of_parameters, game_rounds):
        game = self.new_game(list_of_parameters)
        cumulative_results = [0] * len(list_of_parameters)
        for i in range(game_rounds):
            results = game.start()
            for i, result in enumerate(results):
                cumulative_results[i] += result
        return max(range(len(list_of_parameters)), key=cumulative_results)

    def self_play(self, parameters, game_rounds, input_shape, output_shape):
        game = self.new_game(parameters)
        for i in range(game_rounds):
            p, v = self.self_play_worker(game)

    def self_play_worker(self,game):
        result = game.start()
        return game.history, result