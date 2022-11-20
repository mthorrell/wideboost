import numpy as np
from wideboost.objectives import squareloss


def test_basic_value():
    X = np.zeros([1, 1])
    B = np.ones([1, 1])*10
    Y = np.ones([1, 1])

    G = (X - Y).dot(B.T)
    H = np.sum(np.square(B))

    O = squareloss.squareloss_gradient_hessian(X, B, Y)
    gmatches = G == O[0]
    hmatches = H == O[1]

    assert gmatches and hmatches

# eventually deprecated


def test_old_v_new():
    np.random.seed(123)
    X = np.random.random([100, 10])
    B = np.random.random([10, 1])
    Y = np.random.random([100, 1])

    g, h, pg, ph = squareloss.squareloss_gradient_hessian(X, B, Y)
    go, ho, pgo, pho = squareloss.squareloss_gradient_hessian_FULLHESSIAN(
        X, B, Y)

    eps = 1e-8
    gmatches = np.max(np.abs(g - go)) < eps
    hmatches = np.max(np.abs(h - ho)) < eps

    assert gmatches and hmatches
