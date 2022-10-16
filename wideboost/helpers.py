import numpy as np


def onehot(Y):
    b = np.zeros((Y.shape[0], Y.max().astype(int)+1))
    b[np.arange(Y.shape[0]), Y.astype(int).flatten()] = 1
    return b
