# Original Plan

I believe the right approach for a paper given the easy swappablity of the
scikit-learn metrics, is one which more generally investigates the notion of
Pairwise Prediction Similarity across repeated, varied model runs, and which is
not pre-committed only to a single metric like the EC (just a pairwise IoU /
Jaccard index) nor pre-committed to examining the errors / error sets alone.
The paper would investigate similarity of both erroneous and correct
predictions (be they residuals, or classification predictions) on a collection
of datasets, and with various typical classifiers and/or regressors. There
would be five main sources of (model) variance:

1. data downsampling
1. feature selection [questionable]
1. training sample perturbation
1. test sample perturbation
1. hyperparameter perturbation

For each source of variance (or combination thereof) and each dataset and
model, we would perform N repeats. Each repeat would choose a fixed test set to
use across r validation runs, and each run would have a single set of
predictions on the test set due to the induction procedure. Pairwise prediction
similaritiy metrics (on residuals, correct predictions, and errors; PPSMs) can
be computed across these r runs, and summarized into single values, so that if
we examine p PPSMs, we get N × p summary values per
variance-induction-procedure/dataset/model combination.

Each variance induction procedure also has a degree / magnitude which can be
varied, and then related to the PPSMs (potentially as a separate "experiment"
in the paper). Simulating datasets with known separation / noise can also
strongly aid in interpreting the meaning of the various PPSMs, and reveal
advantages / disadvantages of particular choices of similarity metrics.

## Feature Selection is a Bad Idea

There is no good way to do it.

Most datasets have 10 or more features. Even at the level of 10 C 4 that is 210
different combinations of features, i.e. there is no way to evaluate models
across all feature combinations, and random feature downsampling would simply
produce such radically-different features each time that there would be no way
to know if differences across runs are just due to the feature subsets.

The only way it would be doable is if we could do feature reduction (UMAP) to
various fixed sizes.

but this is not a great idea in general,
since PCA
features to

# Specifics / ALternatives

We could choose *binned* options for each source of variance:

1. data downsampling in [0.30, 0.45, 0.60, 0.75, 0.90] (5)
2. feature selection in []