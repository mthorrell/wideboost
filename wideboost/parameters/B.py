import numpy as np


def RB(nrow,ncol,norm=False):
    B = np.random.random([nrow,ncol])
    if norm:
        B = B / np.sum(B,axis=0)
    return np.random.random([nrow,ncol])

def IB(nrow,ncol,norm=False):
    if(ncol > nrow):
        B = np.concatenate([np.eye(nrow),np.random.random([nrow,ncol-nrow])],axis=1)
    else:
        B = np.concatenate([np.eye(ncol),np.random.random([nrow-ncol,ncol])],axis=0)

    if norm:
        B = B / np.sum(B,axis=0)

    return B

def initialize_B(btype,nrow,ncol):
    output_dict = {
        'R':RB(nrow,ncol),
        'I':IB(nrow,ncol),
        'Rn':RB(nrow,ncol,True),
        'In':IB(nrow,ncol,True)
    }
    return output_dict[btype]



