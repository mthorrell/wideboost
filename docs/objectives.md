Objective Functions
===================

The objective functions listed here are available for both the XGBoost and LightGBM backends.  We follow the naming conventions from both packages. When we provide an objective not available in a package currently (for example, multivariate squared error), we try to extend the naming conventions from either backend to cover this objective.

## Univariate Squared Error


* XGBoost - `'objective':'reg:squarederror'`
* LightGBM - `'objective':'regression'`

:::wideboost.objectives.squareloss.squareloss_gradient_hessian

## Multivariate Squared Error

* XGBoost - `'objective':'multi:squarederror'`
* LightGBM - `'objective':'multiregression'`

:::wideboost.objectives.squareloss.multi_squareloss_gradient_hessian

## Binary Classification

* XGBoost - `'objective':'binary:logistic'`
* LightGBM - `'objective':'binary'`

:::wideboost.objectives.binarylogloss.binarylogloss_gradient_hessian

## Multicategory Classification

* XGBoost - `'objective':'multi:softmax'`
* LightGBM - `'objective':'multiclass'`

:::wideboost.objectives.categoricallogloss.categoricallogloss_gradient_hessian