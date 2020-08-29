from ..wrappers.wxgb import wxgb
from ..wrappers.wlgb import wlgb
import shap
import numpy as np
from copy import deepcopy

class WTreeExplainer:
    def __init__(self,model):
        if type(model) == wxgb:
            self.explainer, self.B = _wxgbExplainer(model)
        if type(model) == wlgb:
            self.explainer, self.B = _wlgbExplainer(model)
    
    def shap_values(self, X):
        n = X.shape[0]
        sv = np.concatenate([x.reshape([-1,1]) for x in self.explainer.shap_values(X)],axis=1).dot(self.B)
        if self.B.shape[1] > 1:
            sv = [sv[:,i].reshape([n,-1]) for i in range(self.B.shape[1])]
        else:
            sv = sv.reshape([n,-1])
        self.expected_value = np.asarray(self.explainer.expected_value).reshape([1,-1]).dot(self.B).flatten()
        return sv 

def _wxgbExplainer(wobject):
    xgbobject = deepcopy(wobject.xgbobject)
    B = wobject.obj.B

    model_bytearray = xgbobject.save_raw()[4:]
    myfun = lambda *args : model_bytearray
    xgbobject.save_raw = myfun

    explainer = shap.TreeExplainer(xgbobject)
    return explainer, B

def _wlgbExplainer(wobject):
    B = wobject.obj.B
    explainer = shap.TreeExplainer(wobject.lgbobject)
    return explainer, B