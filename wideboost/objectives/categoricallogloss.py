import numpy as np
from .general_gh import f_gradient_B, f_hessian_B, row_diag
from ..helpers import onehot

# Use the more general formula, but using the specific formula
# in categoricallogloss_gradient_hessian gives better performance


def categoricallogloss_gradient_hessian_FULLHESSIAN(X, B, Y):
    Y = Y.reshape([Y.shape[0], -1])

    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    logits = X.dot(B)
    max_logit = np.max(logits, axis=1, keepdims=True)
    logits = logits - max_logit

    sum_exp = np.sum(np.exp(logits), axis=1, keepdims=True)
    P = np.exp(logits) / sum_exp

    # Y should already be onehotted
    G = P - Y
    H = row_diag(P) - np.matmul(
        P.reshape([-1, P.shape[1], 1]),
        P.reshape([-1, 1, P.shape[1]])
    )
    flat_H = np.diagonal(H, axis1=1, axis2=2)

    dX = f_gradient_B(G, B)
    dX2 = f_hessian_B(H, B)

    eps = 1e-16
    return dX, np.maximum(dX2, eps), G, flat_H


def categoricallogloss_gradient_hessian(X, B, Y):
    Y = Y.reshape([Y.shape[0], -1])

    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    logits = X.dot(B)
    max_logit = np.max(logits, axis=1, keepdims=True)
    logits = logits - max_logit

    sum_exp = np.sum(np.exp(logits), axis=1, keepdims=True)
    P = np.exp(logits) / sum_exp

    eps = 1e-16
    # Y shold already be onehotted
    G = (P - Y)
    # Not ideal duplication here
    H = row_diag(P) - np.matmul(
        P.reshape([-1, P.shape[1], 1]),
        P.reshape([-1, 1, P.shape[1]])
    )
    flat_H = np.diagonal(H, axis1=1, axis2=2)
    #####
    dX = G.dot(B.transpose())
    dX2 = np.maximum(P.dot(np.square(B).transpose()) -
                     np.square(P.dot(B.transpose())), eps)

    return dX, dX2, G, flat_H
