import numpy as np
from .general_gh import f_gradient_B, f_hessian_B, row_diag

def categoricallogloss_gradient_hessian(X,B,Y):
    Y = Y.reshape([Y.shape[0],-1])
    
    def _onehot(Y):
        b = np.zeros((Y.shape[0], Y.max().astype(int)+1))
        b[np.arange(Y.shape[0]),Y.astype(int).flatten()] = 1
        return b
    
    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2
    
    logits = X.dot(B)
    max_logit = np.max(logits,axis=1,keepdims=True)
    logits = logits - max_logit
    
    sum_exp = np.sum(np.exp(logits),axis=1,keepdims=True)
    P = np.exp(logits) / sum_exp
    
    G = P - _onehot(Y)
    H = row_diag(P) - np.matmul(
        P.reshape([-1,P.shape[1],1]),
        P.reshape([-1,1,P.shape[1]])
    )

    dX = f_gradient_B(G,B)
    dX2 = f_hessian_B(H,B)
    
    eps = 1e-16
    return dX, np.maximum(dX2,eps)

def categoricallogloss_gradient_hessian_old(X,B,Y):
    Y = Y.reshape([Y.shape[0],-1])
    
    def _onehot(Y):
        b = np.zeros((Y.shape[0], Y.max().astype(int)+1))
        b[np.arange(Y.shape[0]),Y.astype(int).flatten()] = 1
        return b
    
    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2
    
    logits = X.dot(B)
    max_logit = np.max(logits,axis=1,keepdims=True)
    logits = logits - max_logit
    
    sum_exp = np.sum(np.exp(logits),axis=1,keepdims=True)
    P = np.exp(logits) / sum_exp
    
    eps = 1e-16
    dX = (P - _onehot(Y)).dot(B.transpose())
    dX2 = np.maximum(P.dot(np.square(B).transpose()) - np.square(P.dot(B.transpose())),eps)
    
    return dX, dX2