# wideboost
Implements wide boosting using popular boosting frameworks as a backend.

## Getting started

```
pip install wideboost
```

## Sample script

### XGBoost back-end

```
import xgboost as xgb
from wideboost.wrappers import wxgb

dtrain = xgb.DMatrix('../../xgboost/demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('../../xgboost/demo/data/agaricus.txt.test')

# Two extra parameters, 'btype' and 'extra_dims'
param = {'btype':'I','extra_dims':2,'max_depth':2, 'eta':0.1, 'objective':'binary:logistic','eval_metric':['error'] }
num_round = 50
watchlist = [(dtrain,'train'),(dtest,'test')]
wxgb_results = dict()
bst = wxgb.train(param, dtrain, num_round,watchlist,evals_result=wxgb_results)
```

## Parameter Explanations
`'btype'` indicates how to initialize the beta matrix. Settings are `'I'`, `'In'`, `'R'`, `'Rn'`.

`'extra_dims'` integer indicating how many "wide" dimensions are used.  When `'extra_dims'` is set to `0` (and `'btype'` is set to `'I'`) then wide boosting is equivalent to standard gradient boosting.

## Reference

Coming Soon!
