from alphazero import optimize, models
from keras.optimizers import SGD

from alphazero import optimize, models, eval_self_play
from multiprocessing import Process, Queue
from logic.mcts_model_based import ModelBasedMCTSLogic
from keras.callbacks import TensorBoard, Callback
import TicTacToe

class SaveOnTrainingEnd(Callback):
    def __init__(self, path):
        super(SaveOnTrainingEnd, self).__init__()
        self.save_path = path

    def on_train_end(self, logs=None):
        self.model.save_weights(self.save_path)

def run():
    width, height, n = 6, 6, 4
    input_shape = (height, width, 2)
    policy_width = width * height

    data_queue, param_queue = Queue(), Queue()

    game_player = eval_self_play.GamePlayer(TicTacToe.Game,
                                            2,
                                            input_shape,
                                            policy_width,
                                            width=width,     # passed to game
                                            height=height,   # passed to game
                                            n_in_row=n)      # passed to game
    best_logic = ModelBasedMCTSLogic(models.new_model(input_shape,policy_width), iterations=100)
    latest_logic = ModelBasedMCTSLogic(models.new_model(input_shape,policy_width), iterations=100)
    p = Process(target = eval_self_play.worker,
                kwargs=dict(game_player=game_player,
                          best_logic=best_logic,
                          latest_logic=latest_logic,
                          data_queue=data_queue,
                          param_queue=param_queue,))
    p.start()

    data = optimize.DataBuffer(input_shape, policy_width, data_len=1000, queue=data_queue)
    model = models.new_model(input_shape, policy_width)
    tensorboard_callback = TensorBoard(write_images=True, write_grads=True)
    save_callback = SaveOnTrainingEnd('latest_model.h5')
    optimize.run(databuffer=data,
                param_queue=param_queue,
                model=model,
                optimizer=SGD(lr=1e-2, momentum=0.9),
                epochs=100,
                 batch_size=32,
                 callbacks=[tensorboard_callback, save_callback])

if __name__=='__main__':
    run()