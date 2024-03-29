Examples
========

## XGBoost Example

```python
import xgboost as xgb
from wideboost.wrappers import wxgb
from pydataset import data
import numpy as np

########
## Get and format the data
DAT = np.asarray(data('Yogurt'))
X = DAT[:,0:9]
Y = np.zeros([X.shape[0],1])
Y[DAT[:,9] == 'dannon'] = 1
Y[DAT[:,9] == 'hiland'] = 2
Y[DAT[:,9] == 'weight'] = 3

n = X.shape[0]
np.random.seed(123)
train_idx = np.random.choice(np.arange(n),round(n*0.5),replace=False)
test_idx = np.setdiff1d(np.arange(n),train_idx)

dtrain = xgb.DMatrix(X[train_idx,:],label=Y[train_idx,:])
dtest = xgb.DMatrix(X[test_idx,:],label=Y[test_idx,:])
#########

#########
## Set parameters and run wide boosting

param = {'btype':'I',      ## wideboost param -- one of 'I', 'In', 'R', 'Rn'
         'extra_dims':10,  ## wideboost param -- integer >= 0
         'max_depth':8,
         'eta':0.1,
         'objective':'multi:softmax',
         'num_class':4,
         'eval_metric':['merror'] }

num_round = 100
watchlist = [(dtrain,'train'),(dtest,'test')]
wxgb_results = dict()
bst = wxgb.train(param, dtrain, num_round,watchlist,evals_result=wxgb_results)
```

## LightGBM Example
```python
import lightgbm as lgb
from wideboost.wrappers import wlgb
from pydataset import data
import numpy as np

########
## Get and format the data
DAT = np.asarray(data('Yogurt'))
X = DAT[:,0:9]
Y = np.zeros([X.shape[0],1])
Y[DAT[:,9] == 'dannon'] = 1
Y[DAT[:,9] == 'hiland'] = 2
Y[DAT[:,9] == 'weight'] = 3

n = X.shape[0]
np.random.seed(123)
train_idx = np.random.choice(np.arange(n),round(n*0.5),replace=False)
test_idx = np.setdiff1d(np.arange(n),train_idx)

train_data = lgb.Dataset(X[train_idx,:],label=Y[train_idx,0])
test_data = lgb.Dataset(X[test_idx,:],label=Y[test_idx,0])
#########

#########
## Set parameters and run wide boosting

param = {'btype':'I',      ## wideboost param -- one of 'I', 'In', 'R', 'Rn'
         'extra_dims':10,  ## wideboost param -- integer >= 0
         'objective':'multiclass',
         'metric':'multi_error',
         'num_class':4,
         'learning_rate': 0.1
        }

wlgb_results = dict()
bst = wlgb.train(param, train_data, valid_sets=test_data, num_boost_round=100, evals_result=wlgb_results)
```

## SHAP Example

```python
from wideboost.explainers.shap import WTreeExplainer
import shap

explainer = WTreeExplainer(bst)
shap_values = explainer.shap_values(data('Yogurt').iloc[0:1,0:9])

shap.initjs()
print(bst.predict(xgb.DMatrix(np.asarray(data('Yogurt'))[0:1,0:9])))
shap.force_plot(explainer.expected_value[3],shap_values[3][0,:],data('Yogurt').iloc[0,0:9])
```
Screenshot of output:
![wideboost-shap](wideboost-shap.png)