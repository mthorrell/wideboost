import numpy as np
from scipy.linalg import lstsq
import xgboost as xgb

from ..helpers import onehot
from ..objectives.squareloss import (
    squareloss_gradient_hessian,
    multi_squareloss_gradient_hessian
)
from ..objectives.categoricallogloss import categoricallogloss_gradient_hessian
from ..objectives.binarylogloss import binarylogloss_gradient_hessian
from ..objectives.manybinarylogloss import manybinarylogloss_all_calcs
from ..evals.classification import (
    error,
    logloss,
    merror,
    mlogloss,
    many_logloss,
    many_auc
)
from ..evals.regression import squarederror, rmse, mae
from ..parameters.B import initialize_B


class wxgb:
    def __init__(self, xgbobject, obj, feval):
        self.obj = obj
        self.feval = feval
        self.xgbobject = xgbobject

    def predict(self, dtrain):
        # TODO: restructure objects to remove duplication
        p = self.xgbobject.predict(dtrain)
        p = p.reshape([p.shape[0], -1])
        output = p.dot(self.obj.B)
        return output


class pred_eval:
    def __init__(self, feval, obj, name, Y2D=None):
        self.feval = feval
        self.obj = obj
        self.name = name
        self.Y2D = Y2D

    def __call__(self, preds, dtrain):
        if not (self.Y2D is None):
            loss = self.feval(preds, dtrain, self.obj, self.Y2D)
        else:
            loss = self.feval(preds, dtrain, self.obj)
        return self.name, loss


def fit(
    X_train, Y_train,
    param, num_boost_round=10, evals=(), obj=None,
    custom_metric=None, maximize=False, early_stopping_rounds=None, evals_result=None,
    verbose_eval=True, xgb_model=None, callbacks=None,
):
    """fit -- Trains a wideboost model using the XGBoost backend. Due to some
    particulars of `xgb.train`, this function cannot accept a DMatrix but must 
    receive the X and Y components separately.

    Args:
        X_train (np.array): X input. 2D.
        Y_train (np.array): Y input. 2D.
        evals (list): List of evaluation datasets. NEEDS to be formatted as
            [((X, Y), 'name'), ...]
        param (dict): Named parameter list. Uses XGBoost conventions. Requires
            two parameters in addition to the usual XGBoost parameters,
            'btype' and 'extra_dims'.

    Returns:
        wxgb: A wxgb object containing the XGBoost object with objective and
            evaluation objects.
    """
    params = param.copy()

    if len(evals) > 1:
        # TODO remove this horrible hack asap
        row_counts = [eval_set[0][0].shape[0] for eval_set in evals]
        assert len(row_counts) == len(set(row_counts)), (
            'evals currently does not work when eval sets have same row counts'
        )

    # Overwrite/set needed params
    if params['objective'] == 'multi:softmax':
        if params.get('num_class'):
            assert params.get('output_dim') is None
            assert Y_train.shape[0] == X_train.shape[0]
            assert len(Y_train.shape) == 1 or Y_train.shape[1] == 1
            params['output_dim'] = params['num_class']
            params['num_class'] = None
            Y_train = onehot(Y_train)

    assert params.get('num_class') is None or params['num_class'] == 1
    assert params['output_dim'] == Y_train.shape[1]
    n_trees_per_round = params.get('output_dim') + params['extra_dims']

    # trick xgb into fitting more trees for us
    dtrain = xgb.DMatrix(
        X_train,
        label=np.zeros(X_train.shape[0] * n_trees_per_round)
    )
    if not (evals is None):
        xgb_evals = _to_xgb_evals(evals, n_trees_per_round)

    if not params.get('beta_eta'):
        params['beta_eta'] = None
    obj = get_objective(params, Y_train)

    params['objective'] = 'reg:squarederror'

    if 'eval_metric' in params:
        custom_metric = get_eval_metric(
            params,
            obj,
            {eval[1]: eval[0][1] for eval in evals}
        )
        params.pop('eval_metric')

    params['disable_default_eval_metric'] = 1

    # TODO: base_score should be set depending on the objective chosen
    if not params.get('base_score'):
        params['base_score'] = 0.0

    params.pop('extra_dims')
    params.pop('output_dim')
    params.pop('btype')
    params.pop('beta_eta', None)

    xgbobject = xgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        evals=xgb_evals, obj=obj, custom_metric=custom_metric, maximize=maximize,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result, verbose_eval=verbose_eval,
        xgb_model=xgb_model, callbacks=callbacks
    )

    return wxgb(xgbobject, obj, custom_metric)


def predict(dtrain, xgbobject, obj):
    P = xgbobject.predict(dtrain)
    P = P.reshape([P.shape[0], -1])
    output = P.dot(obj.B)
    return output


def get_eval_metric(params, obj, Y2D=None):
    output_dict = {
        'error': pred_eval(error, obj, 'error', Y2D),
        'logloss': pred_eval(logloss, obj, 'logloss', Y2D),
        'merror': pred_eval(merror, obj, 'merror', Y2D),
        'mlogloss': pred_eval(mlogloss, obj, 'mlogloss', Y2D),
        'squarederror': pred_eval(squarederror, obj, 'squarederror', Y2D),
        'rmse': pred_eval(rmse, obj, 'rmse', Y2D),
        'mae': pred_eval(mae, obj, 'mae', Y2D),
        'many_logloss': pred_eval(many_logloss, obj, 'many_logloss', Y2D),
        'many_auc': pred_eval(many_auc, obj, 'many_auc', Y2D),
    }
    if (
        isinstance(params['eval_metric'], list)
        or isinstance(params['eval_metric'], tuple)
    ):
        return output_dict[params['eval_metric'][0]]
    return output_dict[params['eval_metric']]


def get_objective(params, Y=None):
    output_dict = {
        'binary:logistic': xgb_objective(
            params['btype'], params['extra_dims'], params['output_dim'],
            binarylogloss_gradient_hessian, params['beta_eta'], Y
        ),
        'reg:squarederror': xgb_objective(
            params['btype'], params['extra_dims'], params['output_dim'],
            squareloss_gradient_hessian, params['beta_eta'], Y
        ),
        'multi:squarederror': xgb_objective(
            params['btype'], params['extra_dims'], params['output_dim'],
            multi_squareloss_gradient_hessian, params['beta_eta'], Y
        ),
        'multi:softmax': xgb_objective(
            params['btype'], params['extra_dims'], params['output_dim'],
            categoricallogloss_gradient_hessian, params['beta_eta'], Y
        ),
        'manybinary:logistic':  xgb_objective(
            params['btype'], params['extra_dims'], params['output_dim'],
            manybinarylogloss_all_calcs,
            params['beta_eta'],
            Y
        )
    }
    return output_dict[params['objective']]


class xgb_objective():
    def __init__(
        self, btype, wide_dim, output_dim,
        obj, beta_eta=None, Y=None
    ):
        # accepted values for obj are the functions
        # associated with
        # "binary:logistic"
        # "reg:squarederror"
        # "multi:softmax"
        # "manybinary:logistic"

        # wide_dim is the number of additional dimensions
        # beyond the output_dim. Can be 0.
        self.wide_dim = wide_dim
        self.output_dim = output_dim
        self.obj = obj
        self.beta_eta = beta_eta
        self.Y = Y

        # B = np.concatenate([np.eye(10),np.random.random([oodim,10])],axis=0)
        # self.B = np.concatenate([np.eye(self.output_dim),
        # np.random.random([self.wide_dim,self.output_dim])],axis=0)
        self.B = initialize_B(btype, self.output_dim +
                              self.wide_dim, self.output_dim)

    def __call__(self, preds, dtrain):
        Xhere = preds.reshape([preds.shape[0], -1])
        if not (self.Y is None):
            Yhere = self.Y
        else:
            Yhere = dtrain.get_label()

        M = Xhere.shape[0]
        N = Xhere.shape[1]

        if not (self.Y is None):
            grad, hess, dp, d2p = self.obj(Xhere, self.B, Yhere)
            if not (self.beta_eta is None or self.beta_eta == 0):
                self.B = update_beta(Xhere, self.B, dp, d2p, self.beta_eta)
        else:
            grad, hess = self.obj(Xhere, self.B, Yhere)

        grad = grad.reshape([M*N, 1])
        hess = hess.reshape([M*N, 1])

        return grad, hess


def update_beta(X, B, G, H, eta):
    assert len(G.shape) == 2
    assert len(H.shape) == 2
    assert G.shape == H.shape
    Y = - G / np.maximum(H, 0.0001)
    newB = lstsq(X, Y)[0]
    return B + eta * newB


def _to_xgb_evals(evals, n_trees_per_round):
    return [
        (
            xgb.DMatrix(
                dataset[0][0],
                label=np.zeros(dataset[0][0].shape[0] * n_trees_per_round)
            ), dataset[1]
        )
        for dataset in evals
    ]
