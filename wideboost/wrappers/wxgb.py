import numpy as np
import xgboost as xgb
from ..objectives.squareloss import squareloss_gradient_hessian, multi_squareloss_gradient_hessian
from ..objectives.categoricallogloss import categoricallogloss_gradient_hessian
from ..objectives.binarylogloss import binarylogloss_gradient_hessian
from ..evals.classification import error, logloss, merror, mlogloss
from ..evals.regression import squarederror, rmse
from ..parameters.B import initialize_B

class wxgb:
    def __init__(self,xgbobject,obj,feval):
        self.obj = obj
        self.feval = feval
        self.xgbobject = xgbobject

    def predict(self, dtrain):
        # TODO: restructure objects to remove duplication
        P = self.xgbobject.predict(dtrain)
        P = P.reshape([P.shape[0],-1])
        O = P.dot(self.obj.B)
        return O

class eval:
    def __init__(self,feval,obj,name):
        self.feval = feval
        self.obj = obj
        self.name = name

    def __call__(self,preds,dtrain):
        loss = self.feval(preds,dtrain,self.obj)
        return self.name, loss

def train(param,dtrain,num_boost_round=10,evals=(),obj=None,
          feval=None,maximize=False,early_stopping_rounds=None,evals_result=None,
          verbose_eval=True,xgb_model=None,callbacks=None):
    """train -- Trains a wideboost model using the XGBoost backend.

    Args:
        param (list): Named parameter list. Uses XGBoost conventions. Requires 
            two parameters in addition to the usual XGBoost parameters,
            'btype' and 'extra_dims'. 

    Returns:
        wxgb: A wxgb object containing the XGBoost object with objective and evaluation objects.
    """
    params = param.copy()
    if not isinstance(obj,xgb_objective):
        assert params['extra_dims'] >= 0
    else:
        print("Using custom objective. Removed extra_dims restriction.")

    # Overwrite needed params
    print("Overwriting param `num_class`")
    try:
        # TODO this is ugly
        nclass = params["num_class"]
    except:
        params['num_class'] = 1

    if isinstance(obj,xgb_objective):
        print("Found custom wideboost-compatible objective. Using user specified objective.")
    else:
        obj = get_objective(params)
    
    params['num_class'] = params['num_class'] + params['extra_dims']
    params.pop('extra_dims')

    print("Overwriting param `objective` while setting `obj` in train.")
    params['objective'] = 'reg:squarederror'

    try:
        feval = get_eval_metric(params,obj)
        params.pop('eval_metric')
        print("Moving param `eval_metric` to an feval.")
    except:
        None

    print("Setting param `disable_default_eval_metric` to 1.")
    params['disable_default_eval_metric'] = 1

    # TODO: base_score should be set depending on the objective chosen
    # TODO: Allow some items to be overwritten by user. This being one of them.
    params['base_score'] = 0.0

    xgbobject = xgb.train(params,dtrain,num_boost_round=num_boost_round,evals=evals,obj=obj,
          feval=feval,maximize=maximize,early_stopping_rounds=early_stopping_rounds,
          evals_result=evals_result,verbose_eval=verbose_eval,xgb_model=xgb_model,
          callbacks=callbacks)

    return wxgb(xgbobject,obj,feval)



def predict(dtrain,xgbobject,obj):
    P = xgbobject.predict(dtrain)
    P = P.reshape([P.shape[0],-1])
    O = P.dot(obj.B)
    return O

def get_eval_metric(params,obj):
    output_dict = {
        'error':eval(error,obj,'error'),
        'logloss':eval(logloss,obj,'logloss'),
        'merror':eval(merror,obj,'merror'),
        'mlogloss':eval(mlogloss,obj,'mlogloss'),
        'squarederror':eval(squarederror,obj,'squarederror'),
        'rmse':eval(rmse,obj,'rmse')
    }
    if isinstance(params['eval_metric'], list) or isinstance(params['eval_metric'], tuple):
        print("Taking first argument of eval_metric. Multiple evals not supported using xgboost backend.")
        return output_dict[params['eval_metric'][0]]
    return output_dict[params['eval_metric']]

def get_objective(params):
    output_dict = {
        'binary:logistic':xgb_objective(params['btype'],params['extra_dims'],params['num_class'],binarylogloss_gradient_hessian),
        'reg:squarederror':xgb_objective(params['btype'],params['extra_dims'],params['num_class'],squareloss_gradient_hessian),
        'multi:squarederror':xgb_objective(params['btype'],params['extra_dims'],params['num_class'],multi_squareloss_gradient_hessian),
        'multi:softmax':xgb_objective(params['btype'],params['extra_dims'],params['num_class'],categoricallogloss_gradient_hessian)
        }
    return output_dict[params['objective']]


class xgb_objective():
    def __init__(self,btype,wide_dim,output_dim,obj):
        ## accepted values for obj are the functions
        ## associated with 
        ## "binary:logistic"
        ## "reg:squarederror"
        ## "multi:softmax"

        ## wide_dim is the number of additional dimensions
        ## beyond the output_dim. Can be 0.
        self.wide_dim = wide_dim
        self.output_dim = output_dim
        self.obj = obj

        #B = np.concatenate([np.eye(10),np.random.random([oodim,10])],axis=0)
        #self.B = np.concatenate([np.eye(self.output_dim),
        #np.random.random([self.wide_dim,self.output_dim])],axis=0)
        self.B = initialize_B(btype,self.output_dim+self.wide_dim,self.output_dim)

    def __call__(self,preds,dtrain):
        Xhere = preds.reshape([preds.shape[0],-1])
        Yhere = dtrain.get_label()

        M = Xhere.shape[0]
        N = Xhere.shape[1]

        grad, hess = self.obj(Xhere,self.B,Yhere)

        grad = grad.reshape([M*N,1])
        hess = hess.reshape([M*N,1])

        return grad, hess
