# Welcome to wideboost!

`wideboost` implements Wide Boosting as described in this [article](https://arxiv.org/abs/2007.09855). `wideboost` does this by wrapping existing popular Gradient Boosting packages ([XGBoost](https://xgboost.readthedocs.io/) and [LightGBM](https://lightgbm.readthedocs.io/)). If you're familiar with those packages `wideboost` can essentially be used as a drop-in replacement for either XGBoost or LightGBM if you're using one of the `wideboost` supported objective functions.

Wide Boosting tweaks the usual Gradient Boosting framework as described in our [Overview](overview.md). It solves the same problems as Gradient Boosting. On several datasets is exhibits much better performance (see the [Overview](overview.md) or the [article](https://arxiv.org/abs/2007.09855)). Supported [objective functions](objectives.md) for `wideboost` include usual univariate and multivariate regression and binary and multi-category classification.

Since Wide Boosting is closely related to Gradient Boosting, we can use the same tools to interpret a `wideboost` model. `wideboost` includes a wrapper on [SHAP](https://github.com/slundberg/shap) to aid in interpreting a `wideboost` model.

* [Installation](installation.md)
* [Overview](overview.md)
* [Examples](examples.md)
* [Supported objective functions](objectives.md)
* [XGBoost wrapper](wrapper_xgboost.md)
* [LightGBM wrapper](wrapper_lightgbm.md)
* [SHAP explainer](shap_explainer.md)

## Reference

```
Horrell, M. (2020). Wide Boosting. arXiv preprint arXiv:2007.09855.
```

