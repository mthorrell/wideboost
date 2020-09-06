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

# from wideboost.objectives.squareloss import squareloss_gradient_hessian as sgh
# from wideboost.objectives.categoricallogloss import categoricallogloss_gradient_hessian as cgh
# import numpy as np

# Y = np.argmax(np.random.multinomial(1,[0.3,0.3,0.4],size = 10),axis=1)
# F = np.random.random([10,4])
# B = np.random.random([4,3])

# GH = cgh(F,B,Y)

# logits = F.dot(B)
# max_logit = np.max(logits,axis=1,keepdims=True)
# logits = logits - max_logit
# sum_exp = np.sum(np.exp(logits),axis=1,keepdims=True)
# P = np.exp(logits)/sum_exp

# gradient = P - _onehot(Y)
# gB = f_gradient_B(gradient,B)
# print(np.abs(gB - GH[0]))

# H = row_diag(P)
# rP = P.reshape([-1,3,1])
# rPT = P.reshape([-1,1,3])
# H = H - np.matmul(rP,rPT)

# hB = f_hessian_B(H,B)
# print(np.abs(hB - GH[1]))
# print(np.max(np.abs(hB-GH[1])))

# def row_diag(M):
#     b = np.zeros((M.shape[0], M.shape[1], M.shape[1]))
#     diag = np.arange(M.shape[1])
#     b[:, diag, diag] = M
#     return b
