
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

Given a feature vector of $n$ samples, $x \in \mathbb{R}^n$, and the pth
percentile $x_p$, then less than or equal to p% of samples fall within
the range $[x_{\min}, x_p]$ (i.e. less than or equal to 5% of samples are in
$[x_{\min}, x_{0.05}]$). Define $\delta = x_p$, and define *the smallest bin
perturbation of size $p$* to be $\tilde{x}$ such that

$$\tilde{x}_i = \min(x_{\min}, x_i + r_i \cdot \delta), \quad r_i \sim \text{U}(-1, 1)$$


#### Perturbation of Categorical Variables

Given a categorical feature $x$ which is a vector of values from the $c$ class
labels $\mathcal{C} = \{0, 1, \dots, c - 1\}$, it is possible to "perturb" $x$
by replacing each label (with some low probability $p$) with a different random label
in $\mathcal{C}$. This would typically be called *label noise* when this kind of
perturbation is on the classification targets, and see  [this
paper](https://doi.org/10.1109/TNNLS.2013.2292894) for an excellent review of
the impacts on all the usual ML classifiers.

However, the impact of *predictor* "label noise" is perhaps less studied. We
thus test all perturbations with 0%, 5%, and 10% predictor label noise.



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
performance to no better than guessing (or even training failure, due to NaNs
resulting from under/overflows). Even within a "good" region of hyperparameter
space, careful tuning of $\theta$ can mean the difference between SOTA and
results that are half a decade out-of-date.

All other considerations being equal, greater sensitivity to $\theta$ means more
time must be spent exploring the hyperparameter space $\Theta$, but that there is
more likely be values in $\Theta$ that lead to increased performance.

### Most Tuning Procedures Assume Fixed Data

Most hyperparameter tuning procedures assume a fixed data $X$
sampled from some fixed distribution $\mathcal{X}$. However, most real-world
deployed models instead are trained on growing data $X_0 \subset X_1 \dots $ or
sequences of datasets in which the training set may only overlap with past training
sets to some limited degree. The frequency and importance of re-tuning will largely
depend on a combination of the sensitivity of the fitting + tuning procedure to
the data $X$ and and the *hyperparmater sensitivity*.


For example, if an algorithm is highly insensitive to the data (e.g.
regularized linear regression like LASSO or or Ridge Regression), one should
expect consistent model performance with updated data *if* $\theta$ is
unchanged. Rather, it is $\theta$ that should be expected to have larger
impacts on future performance. Roughly $|\nabla_X \mathcal{L}_T(X; \theta) | <
|\nabla_\theta \mathcal{L}_T(X; \theta)| $ for tuning procedure T and tuning
loss function $\mathcal{L_T}$.

#### A Definition of Hyper-parameter Sensitivity

Given a model $f$ with parameters $\theta \in \mathbb{R}^m$ that operates on
data $X$ with targets $Y$, with performance criterion $\mathcal{C}$, we can define the overall
model performance to be

$$ \mathcal{C} \big(f(X; \theta), Y\big) \in \mathbb{R}$$


We may not in general have differentiability here with respect to the data $X$,
i.e. neither $\nabla_X\mathcal{C}$ nor $\nabla_\theta\mathcal{C}$ need exist. Or, we may have that
$\theta = (\vartheta, w)$, where $w$ is something like model weights, and where
$\nabla_w\mathcal{C} \big(f(X, \vartheta, w), Y\big)$ exists, but
$\nabla_{\vartheta}$ does not (e.g. the cases where $\vartheta$ is a
categorical or ordinal variable), or is not easily computable. In this case we
tend to call $\vartheta$ the hyperparameters (or hparams, for short).

Since we do *not* generally have gradients for hparams, we need a way to
define hyperparameter sensitivity. Let us propose two notions of hparam
sensitivity

##### Global Hyperparameter Sensitivity

Define the global hyperparameter sensitivity for a model $f$ on data $X$ with criterion $\mathcal{C}$ to be

$$
\eta_{\max}(f, X, \Theta_{\text{def}}) =
\sup_{\theta_1, \theta_2 \in \Theta_{\text{def}}} \Big\lvert \mathcal{C}(f(X; \theta_1)) - \mathcal{C}(f(X; \theta_2)) \Big\rvert
$$

where $\Theta_{\text{def}}$ is some default subset of the total hyperparameter
space widely regarded as containing appropriate hyperparameter ranges, and
where $\lvert \ \cdot \ \rvert$ denotes the absolute value. That is, the global
hparam sensitivity is the largest possible performance difference on
$\Theta_{\text{def}}$.

E.g. if we have a deep-learning classifier model $f$ that classifies $c$
classes, and  $\Theta = \mathbb{R}^2$ and $\Theta_1$ is the deep-learning
(initial) learning rate, and $\Theta_2$ is the deep learning weight decay, then
$\Theta_{\text{def}}$ is (conservatively) something like $\Theta_{\text{def}} =
\Theta_1 \times \Theta_2 = [1 \times 10^{-6}, 1.0] \times [0, 0.1]$, and if
$\mathcal{C}$ is the accuracy, then typically accuracy will vary between
guessing ($1/c$) and something less than perfect accuracy, so that $\eta_{\max}
< \frac{c-1}{c}$.

##### Local Hyperparameter Sensitivity

Suppose we further restrict $\Theta_{\text{def}}$ to be compact (which is
always true in practice), and assume, for the given data $(X, Y)$, that
$\mathcal{C}(\theta) = \mathcal{C}(f(X; \theta), Y)$ is bounded and
continuous (that is, assume for a moment that there are no oridinal or
categorical hyperparameters, and that the real-valued hyperparameters influence
the criterion in a reasonably-well-behaved manner). Then $\mathcal{C}_{\theta}$ is Lipschitz on
$\Theta_{\text{def}}$ and has Lipschitz constant $\eta_{\min}$ such that

$$
\lvert \mathcal{C}(\theta_1) - \mathcal{C}(\theta_2) \rvert
<
\eta_{\min} \left\lVert \theta_1 - \theta_2 \right\rVert
\quad \forall \theta_1, \theta_2  \in \Theta_{\text{def}}
$$

where

$$
\eta_{\min} = \inf_{\eta \in \mathbb{R}} \big\{
\lvert \mathcal{C}(\theta_1) - \mathcal{C}(\theta_2) \rvert
<
\eta \left\lVert \theta_1 - \theta_2 \right\rVert
\;\; \forall \; \theta_1, \theta_2  \in \Theta_{\text{def}} \big\}
$$

Here, $\eta_{\min}$ is the ***local hyperparameter sensitivity*** of $f$ on $X$
for criterion $\mathcal{C}$. In practice, we will never find $\eta_{\min}$, and
given a set of evaluated choices $\Theta_{\text{eval}} = \{\theta_1, \dots, \theta_H\}$, we will simply
define the empirical local hyperparameter sensitivity $\eta_{\min}$ to be

$$
\eta_{\min} \le \max \Bigg\{
\frac{\big\lvert \mathcal{C}(\theta_1) - \mathcal{C}(\theta_2) \big\rvert}
{\lVert \theta_1 - \theta_2 \rVert}
\quad \forall \ \theta_1, \theta_2  \in \Theta_{\text{eval}} \Bigg\}
$$

ie. the empirically largest observed ratio between the criterion distance and
hyperparameter distance is only ever a lower bound on the local hyperparameter
sensitivity.

###### Practical Considerations


We have not yet carefully specified the meaning of $\lVert\cdot\rVert$,
but it should, in most practical cases, be taken to be the L1 or L2 norm.


Differing scales of the continuous hyperparameters can also be a problem for
interpretation of the local hparam sensitivity. E.g. while viable learning
rates for SGD vary in extreme cases from e.g.  $1 \times 10^{-8}$ to $1.0$, a
parameter like the dropout rate varies in practice (also in extreme cases) only
within something like $[0, 0.8]$.

In such cases, rather than using $\lVert \theta_1 - \theta_2 \rVert$, it is likely
prudent to normalize each dimension of $\theta$ to be in $[0, 1]$, or otherwise
log-scale such dimensions so that $\theta_i^{\prime} - \theta_i$ means
something broadly similar to $\theta_j^{\prime} - \theta_j$.


###### Extension to Categorical and Ordinal Hyperparams

For functions of categorical and ordinal parameters, the notion of continuity
makes little sense. However, the definition of the Lipschitz constant can be
extended in practically-useful way quite trivially. We define an ordinal
hyperparameter to be a variable in $\{0, 1, ..., \ell - 1\}$ where $\ell$ is
the number of levels of the ordinal, and a categorical hyperparameter of $c$
possible categories to be a variable which can take on the values in $\{0, 1,
..., c - 1\}$.

Decompose the hyperparameters $\theta$ into $\theta = (\vartheta, \omega,
\kappa)$ for continous hparams $\vartheta \in \mathbb{R}^n$, ordinal hparams
$\omega \in \mathbb{Z}^m$, and categorical hparams $\kappa \in \mathbb{Z}^p$,
such that $n + m + p = N$. Also normalize $\vartheta$ so that $\vartheta \in
[0, 1]^n$ on $\Theta_{\text{def}}$^[Ideally, if components of $\vartheta$ are
typically varied by orders of magnitude, such as the learning rate, then they
should be transformed first (e.g. via logarithm) to keep all components on a
similar scale, and facilitate comparison among different hparam sensitivities].

For $x, y \in \mathbb{R}^n$, define

$$
\left\lVert x- y \right\rVert_{\mathcal{D}} = \sum_i \lvert x_i - y_i \rvert_{\mathcal{D}}
$$

to be the sum of distances of the components of $x$ and $y$. Then define


$$
\lvert x - y \rvert_{\mathcal{D}} =
\begin{cases}
% \lVert \theta^{(i)}_1 - \theta^{(i)}_2 \rVert  & \text{ if } \theta^{(i)} \text{ continuous}\\
\lvert x - y \rvert   & \text{ if } x,y \text{ continuous } \\
\lvert x - y \rvert / \ell  & \text{ if } x,y \text{ ordinal with } \ell \text{ levels }\\
|x \ominus y|/c  & \text{ if } x,y \text{ categorical with } c \text{ classes }\\
\end{cases}
$$

Where $\ominus$ is the symmetric difference $ x \ominus y = (x \cup y)
\setminus (x \cap y)$, such that $|x \ominus y| / c = 1 - |x \cap y|/c$ is just
the proportion of different members of $x$ and $y$. That is, $|\cdot|_{\mathcal{D}}$
just dispatches to whatever normalized distance-like quantity is most appropriate given
the nature of the variable it is applied to.

Then we can define the **normalized local hyperparameter sensitivity** to be

$$
\eta = \max \Bigg\{
\frac{\big\lvert \mathcal{C}(\theta_1) - \mathcal{C}(\theta_2) \big\rvert}
{\mathcal{D}( \theta_1, \theta_2 )}
\quad \forall \ \theta_1 \ne \theta_2  \in \Theta_{\text{eval}}
\Bigg\}
$$

$$
\begin{align}
\mathcal{D}( \theta_1, \theta_2 )
&= \frac{1}{N}
\big(
\left\lVert \theta_1 - \theta_2 \right\rVert_{\mathcal{D}}
\big) \\
&= \frac{1}{N}
\big(
\lVert \vartheta - \vartheta_2 \rVert_{\mathcal{D}} +
\lVert \omega_1 - \omega_2 \rVert_{\mathcal{D}} +
\lVert \kappa_1 - \kappa_2 \rVert_{\mathcal{D}}
\big) \\
\end{align}
$$

that is, it is the largest observed ratio between the difference in the criterion
and the normalized differences between hyperparmeters. Note that
$\mathcal{D}(\theta_1, \theta_2) \in [0, 1]$ regardless of the number of hyperparameters,

Despite notation, this is an intuitive definition: suppose
$\mathcal{C}(\theta)$ is the accuracy and that $\vartheta = \texttt{lr} \in \mathbb{R}$ is
the learning rate, $\omega = \texttt{d} \in \mathbb{Z}$ is the network depth, and $\kappa
= \texttt{opt} \in \{\text{AdamW}, \text{SGD}\}$ is the optimizer. Suppose we also have the
following table of results:

```
   lr    depth      opt     acc
-----  -------  -------  ------
1e-04       34  AdamW    0.2424
1e-04       34  SGD      0.2433
1e-03       34  AdamW    0.2465
1e-03       34  SGD      0.2475
1e-04       50  AdamW    0.2624
1e-04       50  SGD      0.2637
1e-03       50  AdamW    0.2684
1e-03       50  SGD      0.2698
1e-04      101  AdamW    0.3261
1e-04      101  SGD      0.3287
1e-05       34  AdamW    0.3314
1e-05       34  SGD      0.3341
1e-02       34  AdamW    0.3355
1e-03      101  AdamW    0.3382
1e-02       34  SGD      0.3382
1e-03      101  SGD      0.3410
1e-05       50  AdamW    0.3932
1e-05       50  SGD      0.3972
1e-02       50  AdamW    0.3992
1e-02       50  SGD      0.4033
1e-01       34  AdamW    0.4244
1e-01       34  SGD      0.4290
1e-01       50  AdamW    0.5301
1e-01       50  SGD      0.5368
1e-05      101  AdamW    0.5904
1e-05      101  SGD      0.5983
1e-02      101  AdamW    0.6025
1e-02      101  SGD      0.6107
1e-01      101  AdamW    0.8667
1e-01      101  SGD      0.8803

```

due to the relation

```py
def acc(lr: float, depth: int, is_adam: bool) -> float:
   max_depth = 101
   effective_depth = max(max_depth, depth) / max_depth
   opt_lr = 3e-4
   effective_lr = -np.abs(np.log10(lr) - np.log10(opt_lr)) / np.log10(opt_lr)
   max_acc = 0.95
   reduction = 0.98 if is_adam else 1.0
   return effective_depth  * reduction * effective_lr * max_acc + 0.2
dfs = []
for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
   for depth in [34, 50, 101]:
      for is_adam in [True, False]:
         dfs.append(
            DataFrame(
               {"lr": lr, "depth": depth, "AdamW": is_adam, "acc": acc(lr_depth, is_adam) },
               index=[0]
            )
         )
print(pd.concat(dfs, axis=0, ignore_index=True))
```




then $\lVert \theta_1 - \theta_2 \rVert \in [0, 1]$ and
where $\Theta^{(i)}_{\max}$ is the largest

then as above,



(continuous function on compact set is Lipschitz: https://math.stackexchange.com/a/2338876)
  we ideally want to say that
something like that a pseudo-Lipshchitz constant $K^{(i)}$ exists for each
hyperparameter component $\theta^{(i)}$ such that


where:



By contrast, we now have strong reason to believe that, *when properly-trained
with sufficient data* deep neural networks (NNs) will be able to approach or
even exceed human performance in a wide variety of domains with well-defined
problems (e.g. image classification or segmentation, audio/text/image
generation, etc). If we know we have already started with sufficient data $X_0$,
and will only be acquiring increasing samples of data $X_i \subset X_{i + 1}$,
then sensitivity to the data might be less of a concern than "proper training".

That is, given two models with *sufficient capacity for the task at
hand*, we should likely prefer the model with lower hyperparameter sensitivity
$|\nabla_\theta \mathcal{L}_T(X; \theta)|$. In other cases, where a model with
reduced capacity (lower potential performance ceiling) is considerably less
hyperparameter sensitive than a higher-capacity model, we might also prefer
the lower-capacity model, either due to tuning costs, or because the greater
tunability / sensitivity might mean that arbtitrary decisions between very
different but performance-equivalent hyperparams $\theta_1^{\star}, \theta_2^{\star}, \theta_3^{\star}$
must be made.

In addition, a model that is both highly data- and hyperparameter-sensitive may
simply not practically be tuneable at all, since most optimization techniques
assume at least some smoothness and insensitivity to small changes in the hyperparameters.


### Hyperparameter Perturbation

Estimating the hypergradient $\nabla_{\theta}\mathcal{L}(x; \theta)$ is currently
an unsolved problem, and is not in general possible in full given categorical and/or
ordinal components of $\theta$. Estimating the hyperparameter sensistivity is by
contrast straightforward, but computationally expensive.

In practice, continuous hparams are reported with one, or even two signficant
digits, at best, and for ordinal hparams of roughly seven or more levels, a
change by one level is likely assumed to result in minor performance
differences. Some categorical hparams can be expected to result in large
performance differences on some problems (e.g. the choice of radial vs. linear
kernal in a support-vector classifier), but in many cases the differences are
also expected to be smaller (deep-learning optimizer, gradient-boosting loss
function).

Thus, while we should *not* in general expect model performance to be robust
to changes in categorical hyperparameters, we should not expect strong
sensitivity to changes in the second / third significant digits of continuous
hparams, nor to changes of ordinal levels of less than about 10% or so of the
total number of levels (e.g. we ought not to expect much performance
differences between about 200 vs. 250 estimators in XGBoost).

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

# Pairwise Prediction Similarity Across Repeats and Runs

We have a notion of Pairwise Prediction Similarity (PPS) across repeated,
varied model runs, and which is not pre-committed only to a single metric like
the EC (just a pairwise IoU / Jaccard index) nor pre-committed to examining the
errors / error sets alone. We investigate similarity of both erroneous and
correct predictions (be they residuals, or classification predictions) on a
collection of datasets, and with various typical classifiers and/or regressors.
There are five main sources of (model) variance:

- train data downsampling
- feature reduction
- training sample perturbation
- test sample perturbation
- hyperparameter perturbation

For each source of variance (or combination thereof) and each dataset and
model, we would perform $N$ ***repeats***. Each repeat *chooses a fixed test*
set to use across $r$ validation runs in each repeat, and each run has a single
set of predictions on the (shared within a repeat) test set due to the variance
induction procedure. PPS metrics (on residuals, correct predictions, and
errors; PPSMs) can be computed across these $r$ runs, and summarized into
single values, so that if we examine $p$ PPSMs, we get $N \times p$ summary
values per variance-induction-procedure/dataset/model combination.

Each variance induction procedure also has a degree / magnitude which can be
varied, and then related to the PPSMs (potentially as a separate "experiment"
in the paper). Simulating datasets with known separation / noise can also
strongly aid in interpreting the meaning of the various PPSMs, and reveal
advantages / disadvantages of particular choices of similarity metrics.

 
# Tuning and Validation Procedure

## k-fold vs. Monte-Carlo

Suppose we wish to vary training-set size. With k-fold, we train on $\frac{k -
1}{k}$ of the data, such that for $k \in \{2, 3, 4, 5, 10\}$ we train on 50%,
66%, 75%, 80%, and 90% of the data, respectively, but also require $k$ model
fits to obtain one cross-validated performance estimate. That is, the budget
varies by orders of magnitude from 50% to 90% training set size when using
$k$-fold validation. In addition, no one really uses $k < 5$, so this generates
only 2 comparison points (or 3 if we increase to $20$-fold).  By contrast, we
can use $k$ Monte-Carlo splits at arbitrary training set size and keep the
budget the same.

However, we also need to compute PPSMs across a number of repeats. To compute
a PPSM, you need multiple predictions on the same set of samples. There are two
options here: use a fixed holdout test set per repeat, or use repeated shuffled
k-fold and "cobble together" validation-partitions to get total predictions on
the full data. In addition, one must consider how to do tuning given the later
PPSM measurement.

That is, given full data $\mathbf{X}_{\text{full}}$, for each repeat $i$ we
must assemble $N$ predictions $\hat{\bm{y}}_r^{(i)} \in \mathbb{R}^{n}$, for $r
= 1, \dots, N$, on fixed set $\mathbf{X}_{\text{test}}^{(i)}$ with $n$ test
samples, and where $\mathbf{X}_{\text{test}}^{(i)}$ is shared across the $r$
runs in each repeat $i$.

### Comparing Across All Combinations

In addition, we would like to ultimately have a table of something like:

| run | repeat | train_size | samp_perturb | hp_perturb | reduction | acc | preds |
|-----|--------|------------|--------------|------------|-----------|-----|-------|
| ... |  ...   |     ...    |    ...       |    ...     |    ...    | ... |[path] |


But where there are no "hidden" factors or confounds that vary with some column
of the above. For example, if using cobbled k-fold to generate identical
within-repeat test sets, the "cobbled" preds are in fact a result of a differing
number of model fits for each different `train_size`.

Likewise, there is an implicit "expected training set similarity" for each
choice of splitting / validation procedure. For $k$-fold, the proportion of
overlapping samples is $\frac{k-2}{k}$, i.e. 0, 0.5, 0.6, 0.8 for 2-, 4-, 5-,
10-fold respectively (50%, 75%, 80%, and 90% training size).  For MC,
probability of being sampled twice is $P_{\text{train}}^2$ (expected overlap is
in terms of area is expected intersection) so you get 0.25, 0.56, 0.64, 0.81
for 50%, 75%, 80%, and 90% training size. I.e. the difference is small at the
usual choices of training set size.

If grouping by `repeat`, there is a problem with using $k$-fold in that the
groups are of different sizes. If we attempt to compute a variance of any
statistic over a group, then since variance (or any other measure of scale)
is more sensitive to sample size, then we have different levels of confidence
about this variance summary depending on the number of folds. MC avoids this
issue.

## Constructing a Shared Test Set *within* Repeats

There are only two sane approaches:

- **random holdout**: randomly sample (without replacement) $1 - P_{\text{train}}$% of the
  data as $\mathbf{X}_{\text{test}}^{(i)}$. The remaining

### Cobbled vs. MC Downsample

Here, $k$-fold simply seems too expensive given that we want multiple repeats.
MC cross-validation allows a fixed budget of runs (say, 10-50) with precise
variation of the training set size. The only downside / cost to MC is that

### Test Set Variation *Across* Repeats

If we perform 10 or 20 repeats, there is a strong case to make for choosing the
repeat test sets via 10- or 20-fold, and little in favor of MCCV.







I think it is worth looking at model variance with and without tuning. It is conceivable
that tuning improves model performance, but has a different effect on model variance.


# Validation Procedure

## k-fold vs. Monte-Carlo

After setting aside a test set $\mathbf{X}_{\text{test}}$ from the full data
$\mathbf{X}$, leaving behind an $\mathbf{X}_{\text{base}}$. We want to obtain
an estimate of the performance on $\mathbf{X}_{\text{test}}$ by using
cross-validation on $\mathbf{X}_{\text{base}}$, but in a way that *allows the
comparison of different training set sizes*. There are two reasonable approaches:

1.

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
   - [0.25, 0.50, 0.75, None] (4)
1. training sample perturbation
   - Continuous perturbation (4-6)
     - HalfNeighbour, SigDigOne?, RelPercent10, Percentile05?
   - Categorical perturbation (3+)
     - (0, "label"), (0.10, "label"), (0.10, "sample)
     - more sample?
1. hparam perturbation (4)
   - SigOne, SigZero, RelPercent10, AbsPercent10
1. training data downsampling (4)
   - [0.25, 0.50, 0.75, 1.00]
1. test sample perturbation (FREE: can re-use same fitted model)
1. test sample downsampling (FREE: can re-use same fitted model)
1. metrics                  (FREE: can re-use same fitted model)

- datasets (40)
- runs     (r)
- repeats  (N)
- metrics  (FREE)

Total =
         40 datasets ×
         5 downsamples ×
         4-6 continuous perturbs ×
         3-5 categorical perturbs ×
         4 hparam perturbs ×
         4 train downsamples ×
         N runs ×
         r repeats
      = (38400 - 96000) Nr

But then times runs / repeats is  (2 to 8) × rN × 10^4, smallest r is like 5-10, smallest N is like 10-20,
so 2 × 5 × 10 e4 = 1e6  to  8 × 10 × 20 e4 =  1.6e7, 16 million. So **between 1 to 16 million 'validations'
of various sizes and timings**.

# References

## Model Drift / Out-of-Distribution Performance / Model Updating / Calibration Drift / Prediction Differences

- "temporal validation" / "geographic validation" [@steyerbergValidationPredictionModels2019; @moonsPrognosisPrognosticResearch2009]
- "calibration drift" [@hickeyDynamicTrendsCardiac2013]


