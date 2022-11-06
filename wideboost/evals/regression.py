import numpy as np
import xgboost as xgboost

# TODO make these independent of xgb dtrain


def squarederror(preds, dtrain, obj, Y2D):
    # TODO remove duplication (we're generating actual predictions multiple places)
    preds = preds.reshape([preds.shape[0], -1])
    y = _get_valid_y(dtrain, Y2D)

    P = preds.dot(obj.B)

    return 1/2 * np.mean(np.square(y - P))


def rmse(preds, dtrain, obj, Y2D):
    return np.sqrt(2*squarederror(preds, dtrain, obj, Y2D))


def mae(preds, dtrain, obj, Y2D):
    preds = preds.reshape([preds.shape[0], -1])
    y = _get_valid_y(dtrain, Y2D)

    P = preds.dot(obj.B)

    return np.mean(np.abs(y - P))


def _get_valid_y(dtrain, Y2D):
    # awful hack
    return [
        Y2D[k] for k in Y2D.keys()
        if Y2D[k].shape[0] == dtrain.num_row()
    ][0]
