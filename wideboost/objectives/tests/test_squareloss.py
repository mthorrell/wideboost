import numpy as np
from wideboost.objectives import squareloss

def test_basic_value():
    X = np.zeros([1,1])
    B = np.ones([1,1])*10
    Y = np.ones([1,1])

    G = (X - Y).dot(B.T)
    H = np.sum(np.square(B))

    O = squareloss.squareloss_gradient_hessian(X,B,Y)
    gmatches = G == O[0]
    hmatches = H == O[1]

    assert gmatches and hmatches







