# Analysis Plan

We investigate ~40 classification datasets from the OpenML database using some
core machine-learning classifiers:

- Gradient-Boosted Decision Trees (XGBoost)
- Support Vector Machine (scikit-learn)
- Multinomial Logistic Regression (scikit-learn)
- MLP? (PyTorch, probably)
- RandomForest? (XGBoost)

The key concept we investigate is **Model Variance**, that is, how the predictions of
a model change due to various sampling and tuning factors. These various factors we
refer to as ***sources of (model) variance***, and include:

- dataset size
- number of features
- sample noise sensitivity
- hyperparameter sensitivity

Some of these sources of variance can cause different impacts depending whether
they operate in the training data or can contribute to model variance only
during training (***training sources of variance***) or on validation data
(***validation sources of variance***). That is:

- Training Sources of Variance
  - training set size
  - training sample noise
  - hyperparameter sensitivity

