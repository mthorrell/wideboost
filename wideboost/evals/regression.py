import numpy as np

# TODO make these independent of xgb dtrain


def squarederror(preds, dtrain, obj):
    # TODO remove duplication
    preds = preds.reshape([preds.shape[0], -1])
    y = _fix_y_dim(preds, dtrain.get_label())
    y = y.reshape([y.shape[0], -1])

    P = preds.dot(obj.B)

    return 1/2 * np.mean(np.square(y - P))


def rmse(preds, dtrain, obj):
    return np.sqrt(2*squarederror(preds, dtrain, obj))


def mae(preds, dtrain, obj):
    preds = preds.reshape([preds.shape[0], -1])
    y = _fix_y_dim(preds, dtrain.get_label())
    y = y.reshape([y.shape[0], -1])

    P = preds.dot(obj.B)

    return np.mean(np.abs(y - P))


def _fix_y_dim(x, y):
    if y.shape[0] > x.shape[0]:
        dim = int(y.shape[0] / x.shape[0])
        y = y.reshape([x.shape[0], dim])
    return y
