import numpy as np
import xgboost as xgb

# TODO move this to XGB dependent file or
# make these calculations independent of XGB knowledge.
def error(preds,dtrain,obj):
    # TODO remove duplication
    preds = preds.reshape([preds.shape[0],-1])
    y = dtrain.get_label()
    y = y.reshape([y.shape[0],-1])
    
    logits = preds.dot(obj.B)
    p = 1*(logits > 0)
    p = p.reshape([p.shape[0],-1])
    
    return ('error', np.mean(np.abs(p-y)>0.5))

def merror(preds,dtrain,obj):
    # TODO remove duplication
    preds = preds.reshape([preds.shape[0],-1])
    y = dtrain.get_label()
    y = y.reshape([y.shape[0],-1])
    
    logits = preds.dot(obj.B)
    p = np.argmax(logits,axis=1)
    
    return ('error', np.mean(np.abs(p-y)>0.5))

def logloss(preds,dtrain,obj):
    # TODO remove duplication
    # TODO take advantage of log exp
    preds = preds.reshape([preds.shape[0],-1])
    
    logits = preds.dot(obj.B)
    p = 1/(1 + np.exp(-logits))
    y = dtrain.get_label()
    
    p = p.reshape([p.shape[0],-1])
    y = y.reshape([y.shape[0],-1])
    
    p = (2 * p - 1) * y + 1 - p
    
    return 'logloss', -np.mean(np.log(p))

def mlogloss(preds,dtrain,obj):
    # TODO remove duplication
    # TODO take advantage of log exp
    preds = preds.reshape([preds.shape[0],-1])
    y = dtrain.get_label()
    y = y.reshape([y.shape[0],-1])

    def _onehot(Y):
        b = np.zeros((Y.shape[0], Y.max().astype(int)+1))
        b[np.arange(Y.shape[0]),Y.astype(int).flatten()] = 1
        return b

    logits = preds.dot(obj.B)

    max_logit = np.max(logits,axis=1,keepdims=True)
    logits = logits - max_logit

    sum_exp = np.sum(np.exp(logits),axis=1,keepdims=True)
    P = np.exp(logits) / sum_exp

    return -np.mean(np.log(np.sum(P * _onehot(y),axis=1)))
