import numpy as np
from sklearn.metrics import roc_auc_score


# TODO move this to XGB dependent file or
# make these calculations independent of XGB knowledge.
def error(preds, dtrain, obj, Y2D):
    # TODO remove duplication
    preds = preds.reshape([preds.shape[0], -1])
    y = _get_valid_y(dtrain, Y2D)

    logits = preds.dot(obj.B)
    p = 1 * (logits > 0)
    p = p.reshape([p.shape[0], -1])

    return np.mean(np.abs(p-y) > 0.5)


def merror(preds, dtrain, obj, Y2D):
    # TODO remove duplication

    preds = preds.reshape([preds.shape[0], -1])
    y = _get_valid_y(dtrain, Y2D)
    logits = preds.dot(obj.B)

    return np.mean(np.abs(
        np.argmax(logits, axis=1) - np.argmax(y, axis=1)
    ) > 0.5)


def logloss(preds, dtrain, obj, Y2D):
    # TODO remove duplication
    # TODO take advantage of log exp
    preds = preds.reshape([preds.shape[0], -1])

    logits = preds.dot(obj.B)
    p = 1/(1 + np.exp(-logits))
    y = _get_valid_y(dtrain, Y2D)

    p = p.reshape([p.shape[0], -1])
    y = y.reshape([y.shape[0], -1])

    p = (2 * p - 1) * y + 1 - p

    return -np.mean(np.log(p))


def mlogloss(preds, dtrain, obj, Y2D):
    # TODO remove duplication
    # TODO take advantage of log exp
    preds = preds.reshape([preds.shape[0], -1])
    y = _get_valid_y(dtrain, Y2D)

    logits = preds.dot(obj.B)

    max_logit = np.max(logits, axis=1, keepdims=True)
    logits = logits - max_logit

    sum_exp = np.sum(np.exp(logits), axis=1, keepdims=True)
    P = np.exp(logits) / sum_exp

    return -np.mean(np.log(np.sum(P * y, axis=1)))


def many_logloss(preds, dtrain, obj, Y2D):
    validY = _get_valid_y(dtrain, Y2D)
    preds = preds.reshape([preds.shape[0], -1])
    logits = preds.dot(obj.B)
    p = 1/(1 + np.exp(-logits))
    p = (2 * p - 1) * validY + 1 - p
    return -np.mean(np.log(p))


def many_auc(preds, dtrain, obj, Y2D):
    validY = _get_valid_y(dtrain, Y2D)
    preds = preds.reshape([preds.shape[0], -1])
    logits = preds.dot(obj.B)
    return roc_auc_score(validY, logits)


def _get_valid_y(dtrain, Y2D):
    # still a hack
    # TODO improve code structure
    eval_idx = dtrain.name
    return Y2D[eval_idx]
