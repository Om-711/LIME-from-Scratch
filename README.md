
# LIME Explanation for ASTGCN

This repository contains a simple implementation of **LIME (Local Interpretable Model-Agnostic Explanations)** designed to work with an ASTGCN model used for PM2.5 prediction.

## What it does

* Adds Gaussian noise to the input features (perturbation)
* Feeds the perturbed data into the model to get predictions
* Fits a weighted linear regression model using those predictions
* Gives feature and node importance scores
* Generates two plots:

  * Node-wise importance
  * Feature-wise importance

## Files Generated

* `Node_wise_importance.png` — barplot of most important nodes
* `Feature_wise_importance.png` — barplot of most important input features

## Requirements

* PyTorch
* NumPy
* scikit-learn
* seaborn
* matplotlib

## How to Use

```python
from lime_explainer import lime_explainer_astgcn

feature_imp, node_imp = lime_explainer_astgcn(model, batch)
```

Make sure your `batch` contains all the required keys (`features`, `adj`, `v`, `theta`, etc.).

## Notes

* This is not a general-purpose LIME — it is tailored for ASTGCN input format.
* You can adjust `kernel_width` and `num_perturbations` as needed.
* Feature names can be modified in the code depending on your dataset.

