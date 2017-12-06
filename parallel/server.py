from multiprocessing.managers import BaseManager
from parallel.manager import Weights
from queue import Queue

class ServerManager(BaseManager): pass

q = Queue()
best_w = Weights()
new_w = Weights()

ServerManager.register('get_data_queue', lambda: q)
ServerManager.register('get_best_weights', lambda: best_w)
ServerManager.register('get_new_weights', lambda: new_w)

manager = ServerManager((ip, port), auth_key)
manager.start()

q_proxy = manager.get_data_queue()
best_w_proxy = manager.get_best_weights()
new_w_proxy = manager.get_new_weights()