import numpy as np
import xgboost as xgboost

# TODO make these independent of xgb dtrain


def squarederror(preds, dtrain, obj):
    # TODO remove duplication (we're generating actual predictions multiple places)
    preds = preds.reshape([preds.shape[0], -1])
    y = dtrain.get_label()
    y = y.reshape([y.shape[0], -1])

    P = preds.dot(obj.B)

    return 1/2 * np.mean(np.square(y - P))


def rmse(preds, dtrain, obj):
    return np.sqrt(2*squarederror(preds, dtrain, obj))


def mae(preds, dtrain, obj):
    preds = preds.reshape([preds.shape[0], -1])
    y = dtrain.get_label()
    y = y.reshape([y.shape[0], -1])

    P = preds.dot(obj.B)

    return np.mean(np.abs(y - P))
