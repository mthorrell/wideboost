import numpy as np

class xgb_objective():
    def __init__(self,wide_dim,output_dim,obj):
        ## accepted values for obj are:
        ## "binary:logistic"
        ## "reg:squarederror"
        ## "multi:softmax"

        ## wide_dim is the number of additional dimensions
        ## beyond the output_dim. Can be 0.
        self.wide_dim = wide_dim
        self.output_dim = output_dim
        self.obj = obj

        #B = np.concatenate([np.eye(10),np.random.random([oodim,10])],axis=0)
        self.B = np.concatenate([np.eye(self.output_dim),np.random.random([self.wide_dim,self.output_dim])],axis=0)

    def __call__(self,preds,dtrain):
        Xhere = preds.reshape([preds.shape[0],-1])
        Yhere = dtrain.get_label()

        M = Xhere.shape[0]
        N = Xhere.shape[1]

        grad, hess = self.obj(Xhere,self.B,Yhere)

        grad = grad.reshape([M*N,1])
        hess = hess.reshape([M*N,1])

        return grad, hess
