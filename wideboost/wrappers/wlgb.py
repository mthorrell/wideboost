import numpy as np
import lightgbm as lgb
from ..objectives.squareloss import squareloss_gradient_hessian, multi_squareloss_gradient_hessian
from ..objectives.categoricallogloss import categoricallogloss_gradient_hessian
from ..objectives.binarylogloss import binarylogloss_gradient_hessian
from ..evals.classification import error, logloss, merror, mlogloss
from ..evals.regression import squarederror, rmse, mae
from ..parameters.B import initialize_B


class wlgb:
    def __init__(self, lgbobject, obj, feval):
        self.obj = obj
        self.feval = feval
        self.lgbobject = lgbobject

    def predict(self, dtrain):
        # TODO: restructure objects to remove duplication
        P = self.lgbobject.predict(dtrain)
        P = P.reshape([P.shape[0], -1])
        O = P.dot(self.obj.B)
        return O


def train(param, train_set, num_boost_round=100, valid_sets=None,
          valid_names=None, fobj=None, feval=None, init_model=None,
          feature_name='auto', categorical_feature='auto',
          early_stopping_rounds=None, evals_result=None, verbose_eval=True,
          learning_rates=None, keep_training_booster=False, callbacks=None):
    """train -- Trains a wideboost model using the lightGBM backend.

    Args:
        param (list): Named parameter list. Uses lightGBM conventions. Requires 
            two parameters in addition to the usual lightGBM parameters,
            'btype' and 'extra_dims'. 

    Returns:
        wlgb: A wlgb object containing the lightGBM object with objective and evaluation objects.
    """
    params = param.copy()
    if not isinstance(fobj, lgb_objective):
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

    if isinstance(fobj, lgb_objective):
        print(
            "Found custom wideboost-compatible objective. Using user specified objective.")
    else:
        fobj = get_objective(params)

    params['num_class'] = params['num_class'] + params['extra_dims']
    params.pop('extra_dims')

    print("Overwriting param `objective` while setting `fobj` in train.")
    params['objective'] = 'regression'

    try:
        feval = get_eval_metric(params, fobj)
        params.pop('metric')
        print("Moving param `metric` to an feval.")
    except:
        None

    lgbobject = lgb.train(params, train_set, num_boost_round=num_boost_round, valid_sets=valid_sets,
                          valid_names=valid_names, fobj=fobj, feval=feval, init_model=init_model,
                          feature_name='auto', categorical_feature='auto',
                          early_stopping_rounds=early_stopping_rounds, evals_result=evals_result,
                          verbose_eval=verbose_eval, learning_rates=learning_rates,
                          keep_training_booster=keep_training_booster, callbacks=callbacks)

    return wlgb(lgbobject, fobj, feval)


def get_eval_metric(params, obj):
    # TODO match all lightgbm names
    output_dict = {
        'binary_error': eval(error, obj, 'binary_error'),
        'binary_logloss': eval(logloss, obj, 'binary_logloss'),
        'multi_error': eval(merror, obj, 'multi_error'),
        'multi_logloss': eval(mlogloss, obj, 'multi_logloss'),
        'squarederror': eval(squarederror, obj, 'squarederror'),
        'rmse': eval(rmse, obj, 'rmse'),
        'mae': eval(mae, obj, 'mae')
    }
    # TODO enable multiple evals. LightGBM supports this, while
    # xgb does not.
    return output_dict[params['metric']]


class eval:
    def __init__(self, feval, obj, name):
        self.feval = feval
        self.obj = obj
        self.name = name

    def __call__(self, preds, dtrain):
        preds = preds.reshape([-1, self.obj.B.shape[0]], order='F')
        loss = self.feval(preds, dtrain, self.obj)
        return self.name, loss, False


def get_objective(params):
    # TODO use all lgb names (objective can be specified by other names)
    output_dict = {
        'binary': lgb_objective(params['btype'], params['extra_dims'], params['num_class'], binarylogloss_gradient_hessian),
        'multiclass': lgb_objective(params['btype'], params['extra_dims'], params['num_class'], categoricallogloss_gradient_hessian),
        'regression': lgb_objective(params['btype'], params['extra_dims'], params['num_class'], squareloss_gradient_hessian),
        'multiregression': lgb_objective(params['btype'], params['extra_dims'], params['num_class'], multi_squareloss_gradient_hessian)
    }
    return output_dict[params['objective']]


class lgb_objective():
    def __init__(self, btype, wide_dim, output_dim, obj):
        # accepted values for obj are the functions
        # associated with
        # "binary:logistic"
        # "reg:squarederror"
        # "multi:softmax"

        # wide_dim is the number of additional dimensions
        # beyond the output_dim. Can be 0.
        self.wide_dim = wide_dim
        self.output_dim = output_dim
        self.obj = obj

        #B = np.concatenate([np.eye(10),np.random.random([oodim,10])],axis=0)
        # self.B = np.concatenate([np.eye(self.output_dim),
        # np.random.random([self.wide_dim,self.output_dim])],axis=0)
        self.B = initialize_B(btype, self.output_dim +
                              self.wide_dim, self.output_dim)

    def __call__(self, preds, dtrain):

        Xhere = preds.reshape([-1, self.B.shape[0]], order='F')
        Yhere = dtrain.get_label()

        M = Xhere.shape[0]
        N = Xhere.shape[1]

        grad, hess = self.obj(Xhere, self.B, Yhere)

        grad = grad.reshape([M*N], order='F')
        hess = hess.reshape([M*N], order='F')

        return grad, hess
