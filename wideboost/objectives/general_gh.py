import numpy as np

def f_gradient_B(gradient,B):
    assert len(gradient.shape) == 2 
    assert len(B.shape) == 2
    return gradient.dot(B.T)

def f_hessian_B(hessian,B):
    assert len(hessian.shape) == 3
    assert len(B.shape) == 2

    # General formula is diag(B H B^T)
    # This formula is implemented and broadcast to each observation (first dimension).
    # You need the row-wise full hessian. This is why that dimension needs to be 3.
    return np.diagonal(np.matmul(np.matmul(hessian,B.T).transpose([0,2,1]),B.T),axis1=1,axis2=2)

# Helper function for those wanting to use f_hessian_B. Takes a 2D matrix and converts 
# to 3D by making diagonal matrices per row with the diagnoal values equal to the values
# in each row.
def row_diag(M):
    b = np.zeros((M.shape[0], M.shape[1], M.shape[1]))
    diag = np.arange(M.shape[1])
    b[:, diag, diag] = M
    return b