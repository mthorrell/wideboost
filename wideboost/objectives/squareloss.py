import numpy as np
from .general_gh import f_gradient_B, f_hessian_B, row_diag

# Using the general form will be less efficient than
# the purpose-built function, squareloss_gradient_hessian


def squareloss_gradient_hessian_FULLHESSIAN(X, B, Y):
    # Loss = 1/2 (Y - X)^2
    Y = Y.reshape([Y.shape[0], -1])

    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    # Take the gradient and hessian of the usual
    # boosting problem and pass through the general
    # converters in general_gh
    G = X.dot(B) - Y
    H = np.ones([Y.shape[0], 1, 1])

    dX = f_gradient_B(G, B)
    dX2 = f_hessian_B(H, B)

    return dX, dX2, G, np.ones([Y.shape[0], 1])


def multi_squareloss_gradient_hessian(X, B, Y):
    # Loss = 1/2 (Y - X)^2
    Y = Y.reshape([Y.shape[0], -1])

    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    # Take the gradient and hessian of the usual
    # boosting problem and pass through the general
    # converters in general_gh
    G = X.dot(B) - Y
    flat_H = np.ones([Y.shape[0], Y.shape[1]])
    H = row_diag(flat_H)

    dX = f_gradient_B(G, B)
    dX2 = f_hessian_B(H, B)

    return dX, dX2, G, flat_H


def squareloss_gradient_hessian(X, B, Y):
    # Loss = 1/2 (Y - X)^2

    Y = Y.reshape([Y.shape[0], -1])

    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    R = Y - X.dot(B)
    dX = -R * B.transpose()
    dX2 = np.ones(Y.shape) * np.square(B).transpose()

    return dX, dX2, -R, np.ones([Y.shape[0], Y.shape[1]])
