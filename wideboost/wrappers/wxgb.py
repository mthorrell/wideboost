import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator
from ..objectives.squareloss import squareloss_gradient_hessian
from ..objectives.categoricallogloss import categoricallogloss_gradient_hessian
from ..objectives.binarylogloss import binarylogloss_gradient_hessian
from ..evals.classification import error, logloss, merror, mlogloss
from ..evals.regression import squarederror, rmse

# Thinnest possible wrapper for sklearn capabilities
class wxgbModel(BaseEstimator):
    def __init__(self,extra_dims=0,max_depth=None, eta=0.1, n_estimators=10,
                 verbosity=None, objective=None, booster=None,
                 tree_method=None, n_jobs=None, gamma=None,
                 min_child_weight=None, max_delta_step=None, subsample=None,
                 colsample_bytree=None, colsample_bylevel=None,
                 colsample_bynode=None, reg_alpha=None, reg_lambda=None,
                 scale_pos_weight=None, base_score=None, random_state=None,
                 missing=np.nan, num_parallel_tree=None,
                 monotone_constraints=None, interaction_constraints=None,
                 importance_type="gain", gpu_id=None,eval_metric="error",
                 validate_parameters=None,**kwargs):
        self.extra_dims = extra_dims
        self.n_estimators = n_estimators
        self.objective = objective
        self.max_depth = max_depth
        self.eta = eta
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.kwargs = kwargs
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters
        self.eval_metric = eval_metric

    def fit(self, X, y):
        dtrain = xgb.DMatrix(data=X, label=y)
        params = self.__dict__.copy()
        self.wxgbObject = train(params,dtrain)
        return self

    def predict(self,X):
        dtest = xgb.DMatrix(X)
        return self.wxgbObject.predict(dtest)

    def score(self,X,y=None):
        dtest = xgb.DMatrix(data=X,label=y)
        preds = self.wxgbObject.xgbobject.predict(dtest)
        return self.wxgbObject.feval(preds,dtest)[1]


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
    
    params = param.copy()
    assert params['extra_dims'] >= 0

    # Overwrite needed params
    print("Overwriting param `num_class`")
    try:
        # TODO this is ugly
        nclass = params["num_class"]
    except:
        params['num_class'] = 1

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
    print("Taking first argument of eval_metric. Multiple evals not supported using xgboost backend.")
    return output_dict[params['eval_metric'][0]]

def get_objective(params):
    output_dict = {
        'binary:logistic':xgb_objective(params['extra_dims'],params['num_class'],binarylogloss_gradient_hessian),
        'reg:squarederror':xgb_objective(params['extra_dims'],params['num_class'],squareloss_gradient_hessian),
        'multi:softmax':xgb_objective(params['extra_dims'],params['num_class'],categoricallogloss_gradient_hessian)
        }
    return output_dict[params['objective']]


class xgb_objective():
    def __init__(self,wide_dim,output_dim,obj):
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
        self.B = np.concatenate([np.eye(self.output_dim),
        np.random.random([self.wide_dim,self.output_dim])],axis=0)

    def __call__(self,preds,dtrain):
        Xhere = preds.reshape([preds.shape[0],-1])
        Yhere = dtrain.get_label()

        M = Xhere.shape[0]
        N = Xhere.shape[1]

        grad, hess = self.obj(Xhere,self.B,Yhere)

        grad = grad.reshape([M*N,1])
        hess = hess.reshape([M*N,1])

        return grad, hess
