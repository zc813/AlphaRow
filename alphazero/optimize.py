from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
from multiprocessing import Queue
import queue as q

class DataBuffer(object):
    def __init__(self, input_shape, policy_width, data_len = 10000, queue=None, x_dtype=np.int8, y_dtype=np.float32):
        self.data_len = data_len
        self.data_buffer = queue or q.Queue()
        self.current_buffer_size = 0
        self.input_shape = input_shape
        self.policy_width = policy_width
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.x, self.y = self.init_data()
        self.n = 0

    def get_data(self):
        new_x, new_y = self.init_data()
        idx = 0
        l = list()
        l.append(self.data_buffer.get(block=True))
        while True:
            try:
                l.append(self.data_buffer.get(block=False))
            except q.Empty:
                break
        while l:
            x, y = l.pop()
            new_x[idx:min(idx+len(x),self.data_len)] = x
            new_y[idx:min(idx + len(y), self.data_len)] = y
            idx += len(x)
            if idx >= self.data_len:
                break
        self.n += idx
        if idx < self.data_len:
            new_x[idx:self.data_len] = self.x[0:self.data_len-idx]
            new_y[idx:self.data_len] = self.y[0:self.data_len-idx]
        if self.n < self.data_len:
            new_x = new_x[0:self.n]
            new_y = new_y[0:self.n]
        return new_x, new_y

    def init_data(self):
        x = np.zeros((self.data_len, *self.input_shape),dtype=self.x_dtype)
        y = np.zeros((self.data_len, self.policy_width + 1), dtype=self.y_dtype)
        return x,y

    def add_data(self, data):
        self.data_buffer.put(data)
        self.current_buffer_size += len(data)
        if self.current_buffer_size > self.data_len * 1.5:
            self.data_buffer.get()

def alphazero_loss(y_true, y_pred):
    z = y_true[:,-1]
    v = y_pred[:,-1]
    pi = y_true[:,:-1]
    p = y_pred[:,:-1]
    loss = K.square(z-v) - K.sum(pi * K.log(K.clip(p,1e-8,1)), axis=-1)
    return loss

def run(databuffer:DataBuffer, param_queue:Queue, model:Model, optimizer, metrics=None, batch_size=32, epochs=10, verbose=1, callbacks=None):
    print('Optimizing started... waiting for data.')
    # gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    model.compile(optimizer=optimizer, loss=alphazero_loss, metrics=metrics)
    while True:
        x, y = databuffer.get_data()
        # x = np.zeros((500, 6, 6, 2), dtype=int)
        # y = np.zeros((500,37), dtype=float)
        print("Got data. Starting model optimizing...", x.shape, y.shape)
        # model.fit_generator(gen.flow(x, y, batch_size=batch_size), databuffer.data_len//batch_size, epochs)
        model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)
        param_queue.put(model.get_weights())
