import numpy as np
from wideboost.objectives import binarylogloss

def test_basic_value():
    X = np.zeros([1,1])
    B = np.ones([1,1])
    Y = np.ones([1,1])

    g,h = binarylogloss.binarylogloss_gradient_hessian(X,B,Y)
    ### g = P - Y
    ### h = P * (1-P)
    P = 0.5

    gmatches = g == P - 1
    hmatches = h == P * (1-P)

    assert gmatches and hmatches