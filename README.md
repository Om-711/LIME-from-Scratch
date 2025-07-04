

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

---

## 🔍 Steps to Make LIME from Scratch

1. **Create Perturbations**
   Generate random points around the original input using Gaussian noise.

2. **Predict with the Model**
   For each perturbed sample, get the model’s output.
   Use Euclidean distance to compute how far each sample is from the original input.
   Calculate weights using a Gaussian kernel based on the distance.

3. **Train a Simple Interpretable Model**
   Use a linear model (like `LinearRegression`) to fit predictions with perturbations as input.
   This helps estimate how important each input feature is for the final prediction.

4. **Plot and Return Results**
   Plot both:

   * Feature-wise importance
   * Node-wise importance
     Return them for further analysis.

---

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
