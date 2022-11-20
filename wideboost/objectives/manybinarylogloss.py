import numpy as np
from .general_gh import f_gradient_B, f_hessian_B


def manybinarylogloss_all_calcs(X, B, Y):

    Y = Y.reshape([Y.shape[0], -1])

    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    logits = X.dot(B)
    P = 1 / (1 + np.exp(-logits))

    G = P - Y
    flat_H = P - np.square(P)
    H = np.eye(Y.shape[1]) * flat_H[:, np.newaxis, :]
    dX = f_gradient_B(G, B)
    dX2 = f_hessian_B(H, B)

    eps = 1e-16
    return dX, np.maximum(dX2, eps), G, flat_H


def manybinarylogloss_gradient_hessian_FULLHESSIAN(X, B, Y):
    dX, d2X, dP, d2P = manybinarylogloss_all_calcs(X, B, Y)
    return dX, d2X, dP, d2P
