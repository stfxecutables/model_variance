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
1. dimension reduction
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

#### "Half-Neighbour" Perturbation

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

However, consider the open ball $B(\bm{x}, \delta)$, where $\delta < \lVert
\bm{x} - \tilde{\bm{x}} \rVert / 2 $. By the triangle inequality, any sample
drawn from this open ball is closer to $\bm{x}$ than to $\bm{x}_{\text{nn}}$.
That is, this ball is contained within the Voronoi cell of $\bm{x}$.
There are only four possibilities within this open ball:

1. $f(\bm{x}_{\text{nn}}) = f(\bm{x})$ and $f(\bm{x}^{\prime}) = f(\bm{x})$ for
   any $\bm{x}^{\prime}$ in $B(\bm{x}, \delta)$ (including unsampled $\bm{x}^{\prime}$)
1. $f(\bm{x}_{\text{nn}}) \ne f(\bm{x})$ but still $f(\bm{x}^{\prime}) = f(\bm{x})$ for
   any $\bm{x}^{\prime}$ in $B(\bm{x}, \delta)$
1. $f(\bm{x}_{\text{nn}}) = f(\bm{x})$ but $f(\bm{x}^{\prime}) \ne f(\bm{x})$ for
   some unsampled $\bm{x}^{\prime}$ in $B(\bm{x}, \delta)$
   - this means the data is insufficiently sampled between $\bm{x}$ and $\bm{x}_{\text{nn}}$
   - equivalently, this means that if the data reside on some lower-dimensional manifold,
     then that manifold is somewhat ill-behaved near these two points
1. $f(\bm{x}_{\text{nn}}) \ne f(\bm{x})$ but $f(\bm{x}^{\prime}) \ne f(\bm{x})$ for
   some unsampled $\bm{x}^{\prime}$ in $B(\bm{x}, \delta)$
   - this is caused by largely the same issues as in case (2)

Thus, if we assume our data is not too badly undersampled, the "half-neighbour"
ball around each sample represents a viable perturbation space. Arguably, this
is quite a conservative perturbation space relative to the entire Voronoi cell
(see https://arxiv.org/pdf/1905.01019.pdf).

We thus consider two perturbation methods: full-strength "half-neighbour"
perturbation, which takes the open ball as above, and half-strength half-neighbour
perturbation, which takes the open ball with $\delta/2$ instead.

#### Significant Digit Perturbation

A typical measurement has some error. The measurement error may be absolute
(non-heteroscedastic) or relative (heteroscedastic). Without repeated measurements,
estimating such error is not generally possible. However, we can exploit a fact of
human psychology that most measurements will come from devices designed by humans,
and that we will tend to calibrate devices and define measurement scales such that
*when measurements are recorded* meaningfully-different numbers can be represented with just
a small handful of significant digits (say, 3-5).

That is, if some measurement procedure produces measurements of the form
$[1.000000023, 1.000000056, 1.000000059]$, and these represent meaningful
differences, we will tend to subtract 1 (or the mean) and rewrite these as $1 +
[2.3e-8, 5.6e-8, 5.9e-8]$, or define a new measurement unit or scale that otherwise
allows for the differences to be written or displayed with a small number of digits.
Even if a set of values has not been prepared in this manner, subtracting the mean
will then tend to make the values follow this rule.

An exception is for data that is distributed exponentially, or according to some power law.
However, for the datasets here, this does not appear to be the case (there are only in <1%
of values for each dataset, extreme outliers, which may or may not indicate power-law
distributions, but at the least, no distributions seem to be visually obviously exponential).

Thus, we can define perturbation of $n > 0$ significant digits as

$$
\tilde{x} = x + r \cdot \delta \quad \text{where} \quad
\delta = 10^{ \cdot \big\lfloor \log_{10} |x| \big\rfloor \cdot 10^{-n} }
\quad \text{and} \quad r \sim \text{U}(-1, 1)
$$

```python
def sig_perturb(x: ndarray, n_digits: int = 1) -> ndarray:
    delta = 10 ** (np.floor(np.log10(np.abs(x))) / 10 ** n_digits)
    return x + delta * np.random.uniform(-1, 1, x.shape)
```

E.g. from `scratch.py` (note that in scientific notation X.abc... the digit
in "X" we consider the "zeroth" significant digit:

```
Original x value: 2.4382e+02
Showing / perturbing at first significant digit
              x: [[2.4e+02]]
perturbed range:  [2.4e+02 2.5e+02]

Showing / perturbing at second significant digit
              x: [[2.44e+02]]
perturbed range:  [2.43e+02 2.45e+02]

Showing / perturbing at third significant digit
              x: [[2.438e+02]]
perturbed range:  [2.428e+02 2.448e+02]
```

That is, most of the time, the perturbed value will appear to be the same as the
original value when viewed with the original precision, i.e. these are perturbations
that will tend to be "human-invisible" for most naive data printing.

#### Smallest Bin Perturbation

Given a feature vector of $n$ samples, $x \in \mathbb{R}^n$, and the 5th
percentile $x_{0.05}$, then less than or equal to 5% of samples fall within
the range $[x_{\min}, x_{0.05}]$.

## Hyperparameter Sensitivity

Any generally useful algorithm $f$ which learns a mapping $f(x) = y$ from data
samples $x \sim \mathcal{X}$ and targets $y$ will be useful in part because it
has hyperparameters that allow it to be tuned to different distributions and/or
tasks. That is, it will in fact be that $f(x) = f(x; \theta)$ for some $\theta
\in \Theta \subset \mathbb{R}^m$. Supposing such an algorithm also has tunable parameters or "weights"
$w \in \mathbb{R}^q$, we can also write $f(x) = f(x; w, \theta)$, and most optimization
or fitting routines for $f$ will optimize only $w$ (usually because estimation of
$\nabla_{\theta}f$ is not computationally efficient or tractable).

For some algorithms, the choice of $\theta$ (or of most values of theta) will
not have a large impact on the performance most problems. In this case,
intuitively, for continuous hyperparmeters, $\nabla_{\theta}f$ or
$\frac{\partial f}{\partial \theta_i}$ is usually small for most $i$, and for
ordinal (e.g. polynomial fitting degree) or categorical (e.g. choice between
Adam vs SGD optimizer) hyperparameters, the differences between these
ordinal/categorical choices is also small.

However, for other algorithms, the choice of $\theta$ is significiant. Choosing
a bad initial learning rate and/or weight decay can reduce some neural networks
performance no better than guessing (or even training failure, due to NaNs
resulting from under/overflows). Even within a "good" region of hyperparameter
space, careful tuning of $\theta$ can mean the difference between SOTA and
results that are half a decade out-of-date.

All other considerations being equal, greater sensitivity to $\theta$ means more
time must be spent exploring the hyperparameter space $\Theta$.

In addition, most hyperparameter tuning procedures assume a fixed data $X$
sampled from some fixed distribution $\mathcal{X}$. However, most real-world
deployed models instead are trained on growing data $X_0 \subset X_1 \dots $ or
sequences of datasets in which the training set may overlap with past training
sets to varying degrees. The frequency and importance of re-tuning will largely
depend on a combination of the sensitivity of the fitting + tuning procedure to
the data $X$ and and the hyperparmater sensitivity:

For example, if an algorithm is highly insensitive to the data (e.g.
regularized linear regression like LASSO or or Ridge Regression), one should
expect consistent model performance with updated data *if* $\theta$ is
unchanged. Rather, it is $\theta$ that should be expected to have larger impacts
on future performance. Roughly $|\nabla_X \mathcal{L}_T(X) | < |\nabla_\theta \mathcal{L}_T(X)| $ for tuning
procedure T and tuning loss function $\mathcal{L_T}$.

By contrast, we have strong reason to beleive that deep CNNs should be able to
perform well on most image classification tasks that most human beings do not
find to be too difficult (e.g. ImageNet and CIFAR-100 classification). In fact,
it is likely such models perform well on tasks humans find more difficult
(person recogntion, tumor identification, etc), and that it is not
*unreasonable* to assume that if you have an image classification task that
humans or expert humans can perform, and which does not require extensive image
pre-processing, then a CNN ought to be able to perform extremely well *provided
there is sufficient data and/or augmentation*.

That is, deep learning models $f_{\theta}$ are such that we have very strong
reasons to believe that, for *most* problems, there is a $\theta^{\star}$ such
that $f_{\theta^{\star}}$ will outperform almost all classical or custom
approaches. I.e. *regardless of the data $X$*, we can probably tune to some performance greater than the best performance of some classical model.

**HOWEVER** the cost of this tuning (and the relative gain) might be such that it is more
econmically / resource efficient to simply use the classical model that does not need
regular re-tuning with incoming data (or needs it only once a year), even if it loses
some negligible amount of performance.

However, for some deep models (e.g. GANs) training is known to be highly temperamental.
If training data $X_2$ differs enough from $X_1$, it is possible your previously-working
GAN might not even converge at all, or just might produce unacceptable resutls in $X_2$.

However, supposing one has a family of models (e.g. WideResNets) deployed, then
as data rolls in, the sensitivity of performance to $\theta$ (e.g. WideResNet depth and width, augmentation, learning rate, weight decay)

 If an algorithm is known to be sensitive to choice of
$\theta$, then it will necessarily be the case that
data.

In the above case, if a tuning procedure $T(\mathcal{F}, X, \theta) = \theta^{\star}$,
where $\mathcal{F} = \{f_{\theta}: \theta \in \Theta\}$. Or equivalently,
$T(\mathcal{F}, X, \theta) = f_{\theta^{\star}}$. This means the tuning procedure
has a performance:

$$
\mathcal{L}(f_{\theta^{\star}}(X_{\text{test}}, y_{\text{test}}))
$$

$$ |\theta_1^{\star} - \theta_2^{\star}|$$



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
    n_features |   n_datasets
 --------------+-------------
    0  -   9   |    7
   10  -  19   |    6
   20  -  49   |   11
   50 -   99   |    3
  100 -  499   |    6
  500 -  999   |    3
        1000+  |    3
```

### Non-Linear Reductions

We reduce datasets to 25%, 50%, 75%, and 100% of original dataset dimensionalities with
[UMAP](https://arxiv.org/abs/1802.03426). There is no point in using e.g. linear
reduction methods which are dated and harmful in almost all real-world prediction problems.



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
