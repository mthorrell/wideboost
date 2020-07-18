import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator
from .wxgb import train

# Thinnest possible wrapper for sklearn capabilities
class wxgbModel(BaseEstimator):
    def __init__(self,extra_dims=0,btype='I',max_depth=None, eta=0.1, n_estimators=10,
                 verbosity=None, objective=None, booster=None,num_class=None,
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
        self.num_class = num_class
        self.btype = btype
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

