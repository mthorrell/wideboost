# wideboost

Implements wide boosting using popular boosting frameworks as a backend. XGBoost supports the most wideboost features currently. Previous versions supported LightGBM, but this has since been deprecated.

## Getting started

```
pip install wideboost
```

## Sample scripts

The examples folder contains sample scripts for regression, binary classification, multivariate classification and multioutput binary classification. Currently xgboost is the only supported backend.

### Starter script

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
Y = wxgb.onehot(Y)

n = X.shape[0]
np.random.seed(123)
train_idx = np.random.choice(np.arange(n),round(n*0.4),replace=False)
test_idx = np.setdiff1d(np.arange(n),train_idx)

xtrain, ytrain = X[train_idx,:], Y[train_idx,]
xtest, ytest = X[test_idx,:],Y[test_idx,]
########

param = {
    'eta':0.1,
    'btype':'I',      ## wideboost param -- one of 'I', 'In', 'R', 'Rn'
    'extra_dims':1,   ## wideboost param -- integer >= -output_dim
    'beta_eta': 0.01, ## wideboost param -- learning rate for B. Can be unstable -- set to 0 to start.
    'output_dim': 4,  ## wideboost param -- Y must be in a 2D format (ie not a vector of categories)
    'objective':'manybinary:logistic',  ## treat response columns as separate binary problems
    'eval_metric':['many_logloss']      ## average binary logloss across columns
}

num_round = 100
watchlist = [((xtrain, ytrain),'train'),((xtest, ytest),'test')]
wxgb_results = dict()
bst = wxgb.fit(xtrain, ytrain, param, num_round, watchlist, evals_result=wxgb_results, verbose_eval=10)
```

## Parameter Explanations

- `'btype'` indicates how to initialize the beta matrix. Settings are `'I'`, `'In'`, `'R'`, `'Rn'`.
- `'beta_eta'` learning rate for the beta matrix. Sometimes unstable. Start with 0.
- `'output_dim'` width of Y. All Y need to be in 2D matrix format and onehotted if doing categorical prediction.
- `'extra_dims'` integer indicating how many "wide" dimensions are used. When `'extra_dims'` is set to `0` (and `'btype'` is set to `'I'` and `'beta_eta' ` is `0`) then wide boosting is equivalent to standard gradient boosting.

## New Objectives

- `'multi:squarederror'` multidimension output regression.
- `'manybinary:logistic'` loss is independent logloss average across response columns

## New Evals

- `'many_logloss'` logloss averaged across response columns
- `'many_auc'` auc averaged across response columns

## Reference

https://arxiv.org/pdf/2007.09855.pdf

Analyses included in the paper are in the examples/paper_examples/ folder.
