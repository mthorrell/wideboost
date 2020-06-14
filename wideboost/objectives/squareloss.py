import numpy as np 

def squareloss_gradient_hessian(X,B,Y):
    # Loss = 1/2 (Y - X)^2
    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    R = Y - X.dot(B)
    dX = -R * B.transpose()
    dX2 = np.ones(Y.shape) * np.square(B).transpose()

    return dX, dX2

## TODO: Multi-dimension Squareloss

#Y = np.zeros([10,2])
#B = 2*np.ones([])




