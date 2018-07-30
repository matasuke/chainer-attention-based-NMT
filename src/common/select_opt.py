import chainer
from chainer import optimizers

def select_optimizer(type_opt, ):
    if type_opt == "Adam":
        optimizer = optimizers.Adam()
    if type_opt = "SGD":
        optimizer = optimizers.SGD()

    return type_opt
