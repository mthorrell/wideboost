SHAP Explainer
==============

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