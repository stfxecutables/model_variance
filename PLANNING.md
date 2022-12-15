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

## Data Perturbation

Probably best to consider this in the context of *robustness* research (e.g.
adversarial robustness), where we have a perturbation maximum
$\mathbf{\ell}_{\infty} = (\ell^{\;1}_{\infty}, \dots, \ell^{\;p}_{\infty})$
for each feature $x_i$, $i \in \{1, \dots, p\}$, and where we examine the
performance (loss $\mathcal{L}$) of our classifier $f$ on perturbed features:

$$ \mathcal{L}(f(\bm{x + \bm{\delta}}, \bm{y})) $$

where $\bm{\delta} = (\delta_1, \dots, \delta_p)$ and $\delta_i \sim
\text{Uniform}(0, \ell^{\;i}_{\infty} = \epsilon_i)$.

This reduces the perturbation problem to finding a suitable algorithm for
choosing $\ell^{\;i}_{\infty}$ across all datasets.

As this

- could be relative ($x = r \cdot x$ for  $r \sim \text{Uniform}(1- \epsilon ,
  1)$ for each sample $x$)
-

### Label-Preserving Data Perturbation Methods


For a data point $\bm{x} \in \mathbb{R}^p$, a label-preserving perturbation
$\mathcal{P}(\bm{x}) = \tilde{\bm{x}} \in \mathbb{R}^p$ is a point that satisfies both:

1. If there is a function $f: \mathbb{R}^p \mapsto \mathbb{R}$ such that $f(\bm{x}) = y$
   returns the correct label $y$ for $\bm{x}$, then $f(\bm{x}) = f(\tilde{\bm{x}})$
2. $\lVert \bm{x} - \tilde{\bm{x}} \rVert$ is "small"

In general, unless we know the true $f$, we cannot necessarily say if condition (1)
actually holds, unless we can define $\mathcal{P}$ in such a way that it is clear
that condition (1) is satisfied. For example, in the case adversarial attacks on images with
$\ell_{\infty} = 1 / 255$, then since such an intensity perturbation is human-imperceptible,
we can confidently say that it will not alter the human-ascribed image label $y$.

However, for arbitrary tabular data, or even embedded tabular data, we cannot
so easily say whether a perturbation is label-preserving. In particular,
perturbation of a categorical predictor (by switch it to another class label,
for example) may not ever be label-preserving in many datasets. Likewise, where
class manifolds are separated by Hausdorff distance $d$, any perturbation in
the wrong direction, and slightly larger than $d$ will not be label-preserving.

However, we also do not know $d$ in general, nor do we know the direction of $d$.
We cannot even in general say that for a sample $\bm{x}$, and nearest-neighbour
$\bm{x}_{\text{nn}}$, that $\bm{x}_{\text{nn}}$ belongs to the same class as $\bm{x}$.

However, consider the open ball $B(\bm{x}, \delta)$, where $\delta < \lVert \bm{x} - \tilde{\bm{x}} \rVert / 2 $. By the triangle inequality, any sample drawn from this open ball is closer to
$\bm{x}$ than to $\bm{x}_{\text{nn}}$. That is, this ball is contained within the Voronoi
cell of $\bm{x}$.

However, there are only three possibilities for the neighbour $\bm{x}_{\text{nn}}$. Let
$V(\bm{x})$ denote the Voronoi cell

1. $f(\bm{x}_{\text{nn}}) = f(\bm{x})$
2.


that can be observed

a point also the point
$\tilde{\bm{x}} = \bm{x} + \bm{\delta}$ for some $\bm{\delta} \in \mathbb{R}^p$,
which has the property that $\lVert\tilde{\bm{x}} - \bm{x}\rVert < \epsilon(\bm{x})$
A *random perturbation method* $\mathcal{P}$ is
an algorithm that produces such a $\bm{\delta}$ for a given $\bm{x}$ i.e.
$\mathcal{P}(\bm{x}) = \bm{\delta}$, and we can define the set of perturbations
of $\bm{x}$ to be $\mathcal{P}_{\bm{x}} = \{\bm{x} + \mathcal{P}(\bm{x})\} : $
i
set of all

if
A perturbation $\bm{\delta}$ is "safe"

One natural choice for perturbation is to perturb each sample to a random
location within its Voronoi cell, i.e. as in https://arxiv.org/pdf/1905.01019.pdf.
However, this is computationally expensive, and assumes that

There are three possible forms for the function $h$ determining the maximum
perturbation $\epsilon_i$ given data $\bm{X} \in \mathbb{R}^{n \times p}$, s.t.
$\bm{X} = (\bm{x}_1, \dots, \bm{x}_p)$:

- $\epsilon_i = h_i(\bm{x}_i)$:
  - $h_i$ determines the max perturbation from the distribution of the feature in isolation
- $\epsilon_i = h_i(\bm{X})$:
  - $h_i$ determines the max perturbation from the entire dataset



## Feature Selection is a Bad Idea

There is no good way to do it.

Most datasets have 10 or more features. Even at the level of 10 C 4 that is 210
different combinations of features, i.e. there is no way to evaluate models
across all feature combinations, and random feature downsampling would simply
produce such radically-different features each time that there would be no way
to know if differences across runs are just due to the feature subsets.

## Feature Reduction

Number of features varies from 4 - 2001:
```
  0  -   9  |   7
 10  -  19  |   6
 20  -  49  |  11
 50 -   99  |   3
100 -  499  |   6
500 -  999  |   3
      1000+ |   3
```

Supposing we reduce to 25%, 50%, 75%, and 100% of original dataset with UMAP, then we
can technically look at a feature-reduced version of each dataset, and can easily plot
the four choices.

If we limit these analyses to a subset (say, all datasets with n_features > 10, or > 50),
then we could do more fine-grained reductions (e.g. every 20%, every 10%). Based on
https://umap-learn.readthedocs.io/en/latest/performance.html, UMAP is extremely fast.
Our worst dataset is devnagari-script (920 000 samples, 1025 features). With 8 cores
on the precision, this is:

```
Took 525.9128837585449 seconds for ndim=256 (~9 minutes)
Took 1467.2435109615326 seconds for ndim=512 (~25 minutes)
```

so likely with the 40/80 cores on Niagara is very fast even in worst case.

# Specifics / ALternatives

We could choose *binned* options for each source of variance, e.g. data
downsampling in [0.30, 0.45, 0.60, 0.75, 0.90], and so on, or could choose
*random magnitudes* in [0.30, 0.90] (actually, quasi-random is better, e.g.
Latin square, Sobol, poisson disc sampling).

# Fit Times

Devnagari-Script data  ( 92 000 samples, 1025 features):
  - RandomForest on one core:    2 minutes
  - GradientBoost on one core:            (20 minutes?)
  - XGBoost (hist) on one core:           (at least 10 minutes)
Skin-Segmentation data (245 000 samples, 4 features)
  - RandomForest on one core:    6 seconds
  - GradientBoost on one core:
  - XGBoost (hist) on one core:
LDPA                   (164 000 samples, 8 features)
  - RandomForest on one core:   37 seconds
  - GradientBoost on one core:   8 minutes
  - XGBoost (hist) on one core: 13 seconds


# Tuning

We should probably compare XGBoost tuned each time to untuned (or tuned once on full dataset).
We would use the same tuning ranges of https://arxiv.org/pdf/2106.11189v1.pdf, i.e.

| Hyperparameter    |    Type    |    Range     | Log scale |
|-------------------|:----------:|:------------:|:---------:|
| eta               | Continuous |  [0.001, 1]  |     Y     |
| lambda            | Continuous | [1e − 10, 1] |     Y     |
| alpha             | Continuous | [1e − 10, 1] |     Y     |
| num_round         |  Integer   |  [1, 1000]   |     -     |
| gamma             | Continuous |   [0.1, 1]   |     Y     |
| colsample_bylevel | Continuous |   [0.1, 1]   |     -     |
| colsample_bynode  | Continuous |   [0.1, 1]   |     -     |
| colsample_bytree  | Continuous |   [0.5, 1]   |     -     |
| max_depth         |  Integer   |   [1, 20]    |     -     |
| max_delta_step    |  Integer   |   [0, 10]    |     -     |
| min_child_weight  | Continuous |  [0.1, 20]   |     Y     |
| subsample         | Continuous |  [0.01, 1]   |     -     |

It is probably not comparing tuning performance of other dumb models (SVC, LR)

## Binned N Combinations


1. data downsampling
   - [0.30, 0.45, 0.60, 0.75, 0.90] (5)
1. feature selection [questionable]
   - [0.25, 0.50, 0.75, 1.0=None]   (4)
1. training sample perturbation     (5-10)
1. test sample perturbation (FREE: can re-use same fitted model)
1. test sample downsampling (FREE: can re-use same fitted model)
1. hyperparameter perturbation      (5-10)

- datasets (40)
- runs     (r)
- repeats  (N)
- metrics  (FREE)

Total = 40 datasets × 5 downsamples × 4 feature selections × 5-10 sample perturbs × 5-10 hp perurbs
      = 20 000 to 80 000 fits (2e4 to 8e4)

But then times runs / repeats is  (2 to 8) × rN × 10^4, smallest r is like 5-10, smallest N is like 10-20,
so 2 × 5 × 10 e4 = 1e6  to  8 × 10 × 20 e4 =  1.6e7, 16 million. So **between 1 to 16 million 'validations'
of various sizes and timings**.
