import numpy as np 

def binarylogloss_gradient_hessian(X,B,Y):
    ## TODO: make more stable through softmax-max transform
    ## and a minimum value for the hessian
    # Loss = log(p) when Y == 1
    # Loss = log(1-p) when Y == 0
    # p = exp(XB)/(1 + exp(XB))
    # 1-p = 1/(1 + exp(XB))

    Y = Y.reshape([Y.shape[0],-1])

    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    # expand p
    logits = X.dot(B)
    #P = np.exp(logits) / (1 + np.exp(logits))
    P = 1 / (1 + np.exp(-logits))
    eps = 1e-16

    dX = (P - Y) * B.transpose()
    dX2 = np.maximum((P * (1-P) * np.square(B).transpose() ), eps)

    return dX, dX2










