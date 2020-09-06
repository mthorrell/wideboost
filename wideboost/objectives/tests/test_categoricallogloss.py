import numpy as np
from wideboost.objectives import categoricallogloss

def test_basic_value():
    X = np.zeros([1,1])
    B = np.zeros([1,10])
    B[0,0] = 1
    Y = np.ones([1,1])*9

    P = 0.1
    G = P
    H = P - P*P

    O = categoricallogloss.categoricallogloss_gradient_hessian(X,B,Y)
    gmatches = G == O[0]
    hmatches = H == O[1]

    assert gmatches and hmatches