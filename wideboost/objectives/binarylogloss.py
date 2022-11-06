import numpy as np
from .general_gh import f_gradient_B, f_hessian_B


def binarylogloss_gradient_hessian_FULLHESSIAN(X, B, Y):
    Y = Y.reshape([Y.shape[0], -1])

    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    logits = X.dot(B)
    P = 1 / (1 + np.exp(-logits))

    G = P - Y
    flat_H = P - np.square(P)
    H = flat_H.reshape([Y.shape[0], 1, 1])

    dX = f_gradient_B(G, B)
    dX2 = f_hessian_B(H, B)

    eps = 1e-16
    return dX, np.maximum(dX2, eps), G, flat_H


def binarylogloss_gradient_hessian(X, B, Y):
    # Loss = log(p) when Y == 1
    # Loss = log(1-p) when Y == 0
    # p = exp(XB)/(1 + exp(XB))
    # 1-p = 1/(1 + exp(XB))

    Y = Y.reshape([Y.shape[0], -1])

    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    logits = X.dot(B)
    P = 1 / (1 + np.exp(-logits))
    eps = 1e-16

    G = P - Y
    flat_H = P - np.square(P)
    dX = G * B.transpose()
    dX2 = np.maximum((flat_H * np.square(B).transpose()), eps)

    return dX, dX2, G, flat_H
