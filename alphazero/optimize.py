from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
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

    def get_data(self, min_amount=1, sample=0):
        new_x, new_y = self.init_data()
        idx = 0
        l = list()
        amount = 0
        while amount < min_amount:
            data = self.data_buffer.get(block=True)
            l.append(data)
            amount += len(data[0])
        while True:
            try:
                l.append(self.data_buffer.get(block=False))
            except q.Empty:
                break
        while l:
            x, y = l.pop()
            if idx+len(x) < self.data_len:
                new_x[idx:idx+len(x)] = x
                new_y[idx:idx+len(y)] = y
                idx += len(x)
            else:
                new_x[idx:self.data_len] = x[:self.data_len-idx]
                new_y[idx:self.data_len] = y[:self.data_len-idx]
                idx = self.data_len
                break
        self.n += idx
        if idx < self.data_len:
            new_x[idx:self.data_len] = self.x[0:self.data_len-idx]
            new_y[idx:self.data_len] = self.y[0:self.data_len-idx]
        if self.n < self.data_len:
            new_x = new_x[0:self.n]
            new_y = new_y[0:self.n]
        if sample > 0 and sample < self.n:
            new_random_x, new_random_y = new_x.copy(), new_y.copy()
            np.random.shuffle(new_random_x)
            np.random.shuffle(new_random_y)
            return new_random_x[:sample], new_random_y[:sample]
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