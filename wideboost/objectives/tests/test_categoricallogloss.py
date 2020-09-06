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

# eventually deprecated
def test_old_v_new():
    np.random.seed(456)
    X = np.random.random([100,10])
    B = np.random.random([10,5])
    Y = np.random.choice(np.arange(5),[100,1])

    g,h = categoricallogloss.categoricallogloss_gradient_hessian(X,B,Y)
    go,ho = categoricallogloss.categoricallogloss_gradient_hessian_old(X,B,Y)

    eps = 1e-8
    gmatches = np.max(np.abs(g - go)) < eps
    hmatches = np.max(np.abs(h - ho)) < eps

    assert gmatches and hmatches