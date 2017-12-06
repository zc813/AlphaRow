from alphazero.optimize import alphazero_loss, DataBuffer
from alphazero.eval_self_play import max_eval_fn
from keras.optimizers import SGD
from alphazero import models, eval_self_play
from multiprocessing import Queue
from parallel.manager import Weights, ParallelObject, register, new_server, new_client
from multiprocessing import Process
from logic.mcts_model_based import ModelBasedMCTSLogic
from keras.callbacks import TensorBoard, Callback
import TicTacToe
from parallel.manager import Weights
import os.path
import numpy as np
import time

width, height, n = 6, 6, 4
input_shape = (height, width, 2)
policy_width = width * height
savetopath = 'latest_model.h5'
mcts_iterations = 150

def self_play_worker(data_queue, in_weights):
    np.random.seed()
    game_player = new_game_player()
    logic = new_logic()
    weights_id = -1
    print('PLAY | Starting self-play...')
    while True:
        if in_weights.get_id() > weights_id:
            logic.model.set_weights(in_weights.get())
            weights_id = in_weights.get_id()
            print("PLAY | Weights updated to no.%d" % weights_id)
        data = game_player.self_play(logic,4)
        print("PLAY | Sending data,", data[0].shape, data[1].shape)
        data_queue.put(data)

def evaluate_worker(out_weights, in_weights, rounds=10, evaluation_fn=None):
    evaluation_fn = evaluation_fn or max_eval_fn
    game_player = new_game_player()
    latest_logic = new_logic(exploring=False)
    best_logic = new_logic(exploring=False)
    weights_id = -1
    print('EVAL | Waiting for weights...')
    while True:
        if in_weights.get_id() <= weights_id:
            time.sleep(0.5)
            continue
        weights_id = in_weights.get_id()
        print('EVAL | Starting evaluating weights %d' % weights_id)
        weights = in_weights.get()
        latest_logic.model.set_weights(weights)
        results = game_player.evaluate(best_logic, latest_logic, rounds)
        if evaluation_fn(results):
            out_weights.update(weights, weights_id)
            best_logic.model.set_weights(weights)
            print('EVAL | Weights updated! New weights win!')
        else:
            print('EVAL | Weights not updated. New weights lose.')

def optimize(databuffer, out_weights, optimizer, metrics=None, batch_size=32, epochs=50, verbose=1, callbacks=None, augment=False):
    model = new_model()
    model.compile(optimizer=optimizer, loss=alphazero_loss, metrics=metrics)
    print('TRAN | Optimizing started... waiting for data.')
    while True:
        x, y = databuffer.get_data(500)
        print("TRAN | Got data. Starting model optimizing...", x.shape, y.shape)
        model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)
        print('TRAN | Training finished (%d epochs). Sending weights.' % epochs)
        out_weights.update(model.get_weights())

def new_game_player():
    game_player = eval_self_play.GamePlayer(TicTacToe.Game,
                                            2,
                                            policy_width,
                                            width=width,     # passed to game
                                            height=height,   # passed to game
                                            n_in_row=n)      # passed to game
    return game_player

def new_model(load_weights=True):
    model = models.new_model(input_shape, policy_width)
    if load_weights and os.path.isfile(savetopath):
        model.load_weights(savetopath)
        # pass
    return model

def new_logic(exploring=True):
    logic = ModelBasedMCTSLogic(new_model(), iterations=mcts_iterations, exploring=exploring)
    return logic

class SaveOnTrainingEnd(Callback):
    def __init__(self, path):
        super(SaveOnTrainingEnd, self).__init__()
        self.save_path = path

    def on_train_end(self, logs=None):
        self.model.save_weights(self.save_path)

if __name__=='__main__':
    data_queue = Queue()
    latest_weights, best_weights = Weights(), Weights()
    register(data_queue=data_queue, latest_weights=latest_weights, best_weights=best_weights)
    server = new_server()
    server.start()

    client = new_client()
    client.connect()
    data_queue = client.data_queue()
    latest_weights = client.latest_weights()
    best_weights = client.best_weights()
    # data_queue = ParallelObject(Queue, client, 'data_queue')
    # latest_weights = ParallelObject(Weights, client, 'latest_weights')
    # best_weights = ParallelObject(Weights, client, 'best_weights')

    # self-play
    num_processes = 1
    for i in range(num_processes):
        p = Process(target=self_play_worker, args=(data_queue, best_weights))
        time.sleep(0.5)
        p.start()

    # evaluation
    p = Process(target=evaluate_worker, args=(best_weights, latest_weights, 20))
    p.start()

    # optimization
    data = DataBuffer(input_shape, policy_width, data_len=5000, queue=data_queue)
    # tensorboard_callback = TensorBoard(write_images=True, write_grads=True)
    save_callback = SaveOnTrainingEnd(savetopath)
    optimize(databuffer=data,
             out_weights=latest_weights,
             optimizer=SGD(lr=1e-2, momentum=0.9, nesterov=True),
             epochs=50,
             batch_size=32,
             verbose=0,
             callbacks=[save_callback])
