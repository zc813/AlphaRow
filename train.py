from alphazero.optimize import alphazero_loss, DataBuffer
from alphazero.eval_self_play import max_eval_fn
from keras.optimizers import SGD, RMSprop
from alphazero import models, eval_self_play
from multiprocessing import Queue
from parallel.manager import Weights, ParallelObject, register_server, register_client, new_server, new_client
from multiprocessing import Process
from logic.mcts_model_based import ModelBasedMCTSLogic
from keras.callbacks import TensorBoard, Callback
import TicTacToe
from parallel.manager import Weights
import os.path
import numpy as np
import time
import argparse
from gpuscheduler import set_gpu_fraction

width, height, n = 6, 6, 4
input_shape = (height, width, 2)
policy_width = width * height
savetopath = 'latest_model.h5'
mcts_iterations = 200
debug = True

def self_play_worker(ip):
    set_gpu_fraction(0)
    data_queue, _, in_weights = start_client(ip)
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
        data = game_player.self_play(logic, 2)
        print("PLAY | Sending data,", data[0].shape, data[1].shape)
        data_queue.put(data)

def evaluate_worker(rounds, evaluation_fn=None, ip=None):
    set_gpu_fraction(0)
    _, in_weights, out_weights = start_client(ip=ip)
    evaluation_fn = evaluation_fn or max_eval_fn
    game_player = new_game_player()
    latest_logic = new_logic(explore_rounds=2)
    best_logic = new_logic(explore_rounds=2)
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
            best_logic.model.save_weights(savetopath)
            print('EVAL | Weights updated! New weights win!', results)
        else:
            print('EVAL | Weights not updated. New weights lose.', results)

def optimize(databuffer, out_weights, optimizer, metrics=None, batch_size=32, epochs=50, verbose=1, callbacks=None, augmentation=True):
    model = new_model()
    model.compile(optimizer=optimizer, loss=alphazero_loss, metrics=metrics)
    print('TRAN | Optimizing started... waiting for data.')
    while True:
        x, y = databuffer.get_data(500, sample=2000)
        # if debug:
        #     # # DEBUG CODE 1
        #     # for i in range(30):
        #     #     print("SELF:")
        #     #     print(x[i,:,:,0])
        #     #     print("OPPONENT:")
        #     #     print(x[i,:,:,1])
        #     #     print("SCORES:")
        #     #     print(y[i,:36].reshape(6,6))
        #     #     print("PROBABILITY", y[i,36])
        #     #     print("----------------------------")
        #
        #     # DEBUG CODE 2
        #     y_policy = np.reshape(y[:3, :-1], (3, height, width))  # (num, height, width)
        #     y_value = y[:3, -1]  # (num,)
        #     x_test = augment(x[:3])
        #     y_policy = augment(y_policy)
        #     y_test = np.insert(y_policy.reshape(len(y_policy), policy_width), policy_width, replicate(y_value), axis=1)
        #     for i in range(24):
        #         print("SELF:")
        #         print(x_test[i,:,:,0])
        #         print("OPPONENT:")
        #         print(x_test[i,:,:,1])
        #         print("SCORES:")
        #         print(y_test[i,:36].reshape(6,6))
        #         print("PROBABILITY", y_test[i,36])
        #         print("----------------------------")
        begin_time = time.time()
        print("TRAN | Got data. Starting model optimizing...", x.shape, y.shape)
        if augmentation:
            y_policy = np.reshape(y[:,:-1],(len(y),height,width)) #(num, height, width)
            y_value = y[:,-1] #(num,)
            x = augment(x)
            y_policy = augment(y_policy)
            y = np.insert(y_policy.reshape(len(y_policy), policy_width), policy_width, replicate(y_value), axis=1)
            print('TRAN | Augmented:', x.shape, y.shape)
        history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, )
        print('TRAN | Training finished (%d epochs). Sending weights. Time: %.2fs' % (epochs, time.time()-begin_time))
        print('TRAN | Loss:', *map(lambda f: format(f, '.3f'),history.history['loss']))
        out_weights.update(model.get_weights())

def augment(x):
    dim = len(x.shape)
    x1 = np.transpose(x, [0,2,1] + [i for i in range(3,dim)])
    return np.concatenate([rotatetofour(x), rotatetofour(x1)], axis=0)

def replicate(x):
    return np.concatenate([x for i in range(8)], axis=0)

def rotatetofour(x):
    return np.concatenate([x, np.flip(x, 1), np.flip(x, 2), np.flip(np.flip(x, 1), 2)], axis=0)

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

def new_logic(explore_rounds=-1):
    logic = ModelBasedMCTSLogic(new_model(), iterations=mcts_iterations, explore_rounds=explore_rounds)
    return logic

class SaveOnTrainingEnd(Callback):
    def __init__(self, path):
        super(SaveOnTrainingEnd, self).__init__()
        self.save_path = path

    def on_train_end(self, logs=None):
        self.model.save_weights(self.save_path)

def start_server():
    set_gpu_fraction(0)
    data_queue = Queue()
    latest_weights, best_weights = Weights(), Weights()
    register_server(data_queue=data_queue, latest_weights=latest_weights, best_weights=best_weights)
    server = new_server()
    server.get_server().serve_forever()

def start_client(ip=None):
    # data_queue = Queue()
    # latest_weights, best_weights = Weights(), Weights()
    ip = ip or '127.0.0.1'
    register_client('data_queue', 'latest_weights', 'best_weights')
    print(ip)
    client = new_client(ip=ip)
    client.connect()
    data_queue = client.data_queue()
    latest_weights = client.latest_weights()
    best_weights = client.best_weights()
    # data_queue = ParallelObject(data_queue, client, 'data_queue')
    # latest_weights = ParallelObject(latest_weights, client, 'latest_weights')
    # best_weights = ParallelObject(best_weights, client, 'best_weights')
    return data_queue, latest_weights, best_weights

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--client', type=str, metavar='IP',
                        help='specify ip address to connect to; if not specified, a new server will be started on localhost')
    parser.add_argument('-p', '--selfplay', type=int, default=3, metavar='N', help='number of self-play workers')
    parser.add_argument('--noevaluate', '-e', action='store_true')
    parser.add_argument('--nooptimize', '-o', action='store_true')
    args = vars(parser.parse_args())
    num_processes = args.get('selfplay')
    ip = args.get('client')
    if ip is not None:
        is_client = True
    else:
        is_client = False
        ip = '127.0.0.1'
    has_evaluator = not args.get('noevaluate')
    has_optimizer = not args.get('nooptimize')

    if not is_client:
        # start new server
        p = Process(target=start_server)
        p.start()
        time.sleep(1.0)

    # self-play
    for i in range(num_processes):
        p = Process(target=self_play_worker, kwargs=dict(ip=ip))
        time.sleep(0.5)
        p.start()

    if has_evaluator:
        # evaluation
        p = Process(target=evaluate_worker, args=(10,), kwargs=dict(ip=ip))
        p.start()

    if has_optimizer:
        # optimization
        # set_gpu_fraction(0.8)
        data_queue, latest_weights, _ = start_client(ip=ip)
        data = DataBuffer(input_shape, policy_width, data_len=20000, queue=data_queue)
        # tensorboard_callback = TensorBoard(write_images=True, write_grads=True)
        save_callback = SaveOnTrainingEnd(savetopath)
        optimize(databuffer=data,
                 out_weights=latest_weights,
                 optimizer=SGD(lr=1e-2, momentum=0.9, nesterov=True),
                #  optimizer=RMSprop(),
                 epochs=30,
                 batch_size=32,
                 verbose=0,
                 callbacks=[])