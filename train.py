from alphazero import optimize, models
from alphazero.eval_self_play import GamePlayer
from keras.optimizers import SGD
from keras.optimizers import SGD

from alphazero import optimize, models
from alphazero.eval_self_play import GamePlayer


def run():
    data = optimize.DataBuffer()
    model = models.new_model()

    optimize.run(data, model, SGD(lr=1e-2, momentum=0.9), metrics=['accuracy'])

def evaluate_and_play():
    gameplayer = GamePlayer()
    data = gameplayer.self_play()
    queue.add(data)

if __name__=='__main__':
    run()