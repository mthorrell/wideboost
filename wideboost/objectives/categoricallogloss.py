import numpy as np

def categoricallogloss_gradient_hessian(X,B,Y):
    ## TODO: Fix so that it is softmax(X - max(x)) for numerical stability
    
    Y = Y.reshape([Y.shape[0],-1])
    
    assert len(X.shape) == 2
    assert len(B.shape) == 2
    assert len(Y.shape) == 2

    logits = X.dot(B)

    Z = np.exp(X.dot(B))

    ZtBi = Z.dot(B.transpose())
    ZtBi2 = Z.dot(np.square(B.transpose()))
    
    Zt1 = np.tile(np.sum(Z,axis=1,keepdims=True),[1,B.shape[0]])

    dX =  ZtBi/Zt1 - B[:,Y.astype(int)].transpose()
    dX2 = ZtBi2 / Zt1 - np.square(ZtBi / Zt1)

    return dX, dX2



def cgh(X,B,Y):
    ## TODO: Verify the numbers are right
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