I've been working on what I call the "model variance" paper quite extensively for
the past weeks, and have preliminary results, and then some final decisions to
make for what is worth running, compute-wise.

# Paper Approach

# Overview

I examine variance in classifier *performance* and performance *consistency*
(collectively, "model / classifier variance") due to "small" perturbations that
can occur in various aspects of the typical tune-train-evaluate analysis
pipeline. This is done by implementing a number of *training perturbation schemes*
which may operate on the training data values, training hyperparameter (hparam)
values, or which simply impact which training samples are selected.
So for example, an hparam perturbation scheme randomly alters the training hparams
in some way.

The primary unit of analysis in this study is the **repeat**. A repeat is a
collection of 10 **runs** wherein each run shares the same *test set*, but where
training is perturbed by a combination of
perturbation schemes. For example, if we have 2 hparam perturbation schemes, 3
data perturbation schemes, and 4 downsampling schemes, then there are (2 +
1)·(3 + 1)·(4 + 1) = 60 total possible combinations (we also always test "no
perturbation" for each category of perturbation scheme, hence the +1).

While the combination of perturbation schemes is fixed for each repeat, each run's training
is unique due to the random training perturbations (the exception being when all perturbations
are none, in which a deterministic classifier will have identical fits). This means
**each repeat yields a collection of 10 predictions** which can than be summarized using
various Pairwise Prediction Similarity Metrics (PPSMs) like the EC, or any other
pairwise distance / similarity metrics (e.g. Cohen's Kappa).

We also perform multiple repeats so that we can get a distribution on the various
metrics above. Each repeat has a different test set, and each repeat and run uses
a different, independently-seeded random number generator, so we expect variance
as well from repeat to repeat.


# Data

I use most of the 39 tabular datasets used in https://arxiv.org/abs/2106.11189.
I exclude two datasets (Fashion MNIST, and DevnagariScript) which are in fact
image datasets, and for 37 datasets total:

| name                                   |   n_sample |   n_feat |   n_majority_cls |   n_minority_cls |   n_cls |
|:---------------------------------------|-----------:|---------:|-----------------:|-----------------:|--------:|
| arrhythmia                             |        452 |      280 |              245 |              207 |       2 |
| Australian                             |        690 |       15 |              383 |              307 |       2 |
| blood-transfusion-service-center       |        748 |        5 |              570 |              178 |       2 |
| vehicle                                |        846 |       19 |              628 |              218 |       2 |
| anneal                                 |        898 |       39 |              486 |              412 |       2 |
| credit-g                               |       1000 |       21 |              700 |              300 |       2 |
| cnae-9                                 |       1080 |      857 |              120 |              120 |       9 |
| car                                    |       1728 |        7 |             1210 |               65 |       4 |
| mfeat-factors                          |       2000 |      217 |             1800 |              200 |       2 |
| kc1                                    |       2109 |       22 |             1783 |              326 |       2 |
| segment                                |       2310 |       20 |              330 |              330 |       7 |
| jasmine                                |       2984 |      145 |             1492 |             1492 |       2 |
| kr-vs-kp                               |       3196 |       37 |             1669 |             1527 |       2 |
| sylvine                                |       5124 |       21 |             2562 |             2562 |       2 |
| phoneme                                |       5404 |        6 |             3818 |             1586 |       2 |
| christine                              |       5418 |     1637 |             2709 |             2709 |       2 |
| fabert                                 |       8237 |      801 |             1927 |              502 |       7 |
| dilbert                                |      10000 |     2001 |             2049 |             1913 |       5 |
| nomao                                  |      34465 |      119 |            24621 |             9844 |       2 |
| jungle_chess_2pcs_raw_endgame_complete |      44819 |        7 |            23062 |             4335 |       3 |
| bank-marketing                         |      45211 |       17 |            39922 |             5289 |       2 |
| adult                                  |      48842 |       15 |            37155 |            11687 |       2 |
| shuttle                                |      58000 |       10 |            45586 |               10 |       7 |
| volkert                                |      58310 |      181 |            12806 |             1361 |      10 |
| helena                                 |      65196 |       28 |             4005 |              111 |     100 |
| connect-4                              |      67557 |       43 |            44473 |             6449 |       3 |
| jannis                                 |      83733 |       55 |            38522 |             1687 |       4 |
| numerai28.6                            |      96320 |       22 |            48658 |            47662 |       2 |
| higgs                                  |      98050 |       29 |            51827 |            46223 |       2 |
| aloi                                   |     108000 |      129 |              108 |              108 |    1000 |
| MiniBooNE                              |     130064 |       51 |            93565 |            36499 |       2 |
| walking-activity                       |     149332 |        5 |            21991 |              911 |      22 |
| ldpa                                   |     164860 |        8 |            54480 |             1381 |      11 |
| skin-segmentation                      |     245057 |        4 |           194198 |            50859 |       2 |
| CreditCardFraudDetection               |     284807 |       31 |           284315 |              492 |       2 |
| Click_prediction_small                 |     399482 |       12 |           332393 |            67089 |       2 |
| dionis                                 |     416188 |       61 |             2469 |              878 |     355 |


# Classifiers

I fit as classifiers: Logistic Regression (LR), linear Support Vector Machine
(SVM), XGBoost (XGB), and a modern-ish middle-sized neural network with
dropout, batch-norm, and etc (essentially the one in [this
paper](https://arxiv.org/pdf/1705.03098.pdf)). This network is just "MLP" here
and in code, and is implemented in PyTorch and requires a GPU to train
efficiently.

The LR and SVM models have to be fit via SGD (technically,
[SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)),
so they are a bit non-standard, but early testing showed the plain LR and SVM
algorithms simply can't handle the larger of the 38 datasets (compute times
will be exorbitant), and that, at least on the smaller or moderate-sized
datasets, the SGD variants had about the same holdout performance anyway (often
actually slightly better).

A radial-basis SVM unfortunately cannot be easily git via SGD. It is
technically possible by using kernel approximation, e.g. via the [Nystroem
transformer](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html),
but this departs heavily from the usual SVM, and introduces significant tuning
and interpretational complexity, so is not used in this study.

# Sources of Variance and Training Perturbation Schemes

A "source of (model) variance" is anything in the training procedure that might
impact the resulting predictions. I explicitly manipulate three main sources of
variance in this study: hyperparameters, data (predictor) noise, and training
sample distribution / size.

## Train Downsampling

Train downsampling by $p$ percent uses only $p$ percent of the available
training samples $\mathbf{X}$. For reasons related to stratification, compute
costs, and sample sizes, I limit $p$ to values in
$\{25\%, 50\%, 75\%, 100\%\}$.

## Data Perturbation

I develop a number of methods to perturb predictor values and simulate
"small" predictor noise. I classify predictor features as either continuous or
categorical, and define perturbation methods based on the cardinality.


### Continous Data Perturbation

These perturbations are all designed to be "small" in various intuitive ways,
and define. For all descriptions below, training data is
$\mathbf{X} \in \mathbb{R}^{N \times F}$, with $N$ samples, and $F$
features, and $f$ is the perturbation function, which may be either
$f: \mathbb{R}^{N \times F} \mapsto \mathbb{R}^{N \times F}$ if it requires
the full data, or simply $f: \mathbb{R} \mapsto \mathbb{R}$ if it operates on
feature values only.

**Significant-digit**: Rewriting each feature sample $x \in \mathbb{R}$ in
scientific notation, e.g. $x =$ `1.2345eN` for some integer `N`, then define
*perturbation at the zeroth digit* to be
$f(x) = x + e, \; e \sim \text{Unif}($`-1eN`, `1eN`$)$. *Perturbation of the first
digit* is similar but takes $e \sim \text{Unif}($`-0.1eN`, `0.1eN`$)$, and perturbation
to the third digit is $e \sim \text{Unif}($`-0.01eN`, `0.01eN`$)$, and so on.

The idea is that this is a perturbation that is "visible" to humans
when looking at rounded tables of data, and that perturbations at a level that
should be mostly invisible to humans (e.g.at the 3rd or 4th significant digit)
should NOT have dramatic impacts on classifier behavior.

**Nearest-Neighbor**: This perturbs each sample $\symbfit{x} \in \mathbb{R}^F$
to within its Voronoi cell, as defined by $\mathbf{X}$.
That is, if $\mathbf{x}_{\text{nn}}$ is the nearest neighbour to $\symbfit{x}$,
and $B(\symbfit{a}, r)$ is the multidimensional ball of radius $r$ centred at
$\symbfit{a}$, then neighbor-based perturbation moves $\symbfit{x}$ to a
random location in $B(x, \; c\cdot \lVert \symbfit{x} - \symbfit{x}_{\text{nn}} \rVert)$,
where $c \in \{0.25, 0.5, 1.0\}$. There is precedence for this in e.g.
https://arxiv.org/pdf/1905.01019.pdf, and the basic reasoning is quite sound.
E.g. at "half" neighbour perturbation (c=0.5) the perturbed value's nearest
neighbour does not change, so a KNN classifier with K=1 would not change its
predictions under this kind of perturbation.

**Relative**: For each feature sample $x \in \mathbb{R}$, define
$f(x) = x + e$, where $e \sim \text{Unif}\big(x (1 - p),\;x  (1 + p)\big)$, for
$p \in \{0.1, 0.2, \dots\}$.

**Percentile**: Each feature has a distribution of *absolute* values, and lower percentile $p$
for $p < 0.5$ for those absolute values.  For each feature sample $x \in \mathbb{R}$, define
$f(x) = \text{clamp}(x + e, x_{\min}, x_{\max})$, where $e \sim \text{Unif}\big(x - p / 2,\; x + p/2\big)$,
where $x_{\min}$ and $x_{\max}$ are the largest observed values for the feature, and where
$p \in \{0.1, 0.2, \dots\}$. We use a percentile on the absolute values to avoid overly-large
perturbations due to extreme negative or extreme positive values.


### Categorical Data Perturbation

**Sample-level**: A proportion $p$ is chosen, and $\lceil p N\rceil$ samples are chosen
to be potentially perturbed. If there are $c$ categorical feature, then with
probability $\frac{1}{c}$, for each sample, each label is set to a random label in
that feature. I.e. we expect, on average, for each sample chosen for perturbation, that only
one of that sample's categorical features gets a random (possible the same) label.
This simulates something like data entry error / noise.

**Label-level**: Let $X_c$ be the $N \times c$ matrix of
categorical predictors. For small probability $p$. Define `idx =
np.random.uniform(0, 1, X_c.shape) < p`. Then the values where idx is True
are replaced with a random label from the available labels for that feature.
That is, this is "label noise" but in the predictors, apparently also more
correctly called ["attribute noise"](https://link.springer.com/article/10.1007/s10462-004-0751-8), and the mixing with a uniform has some resemblance to label smoothing in deep learning.


**Note**: Early experiments show that sample-level perturbation is an extremely
weak source of variance even at $p = 0.2$ (does not seem to impact performance
distributions at all).  By contrast, label-level perturbation at even $p=0.1$ has obvious performance impacts.
However, label-level perturbation is systemic and aggressive, so more likely appropriate values
for this are $p \le 0.1$, then




## Hyperparameter (Hparam) Perturbation

Hyperparameters can be categorized based on their cardinality: they are either continuous,
ordinal, or categorical. They are also either logarithmic or not. For the most part,
hyperparameters can be perturbed in the same ways that the data (i.e. predictors $\mathbf{X}$)
are perturbed, and so we define *significant digit* and *relative* hparam perturbation
of continous and ordinal hparams as above. However, since we must specify viable ranges for
hparams, and don't have to worry about outliers, we can just use proportions of the
full hparam range instead of percentiles, and so all this *absolute* perturbation.


When a param is logarithmic, perturbations are done instead in
$\log_{10}$-space, and then converted back. E.g. if we have tuned the model and
found some good logarithmic parameter $\theta \in \mathbb{R}$, and are
perturbing relatively by $p = 0.1$, then, the pertubed value is $\theta + 10^e$, where
$e \sim \text{Unif}\big((1-p)\log_{10}\theta,\; (1 + p)\log_{10}\theta\big)$.

Ordinals are treated as continuous, but with rounding, as needed, to ensure
values stay within the original ordinal levels. Categorical hparams are
perturbed independently with probability $p$.

**Note**: Currently I forget to scale the perturbation probability for categoricals
to be proportional to the number of categorical hparams and runs. The categorical perturbation
prob is set to be in $\{0.1, 0.2\}$. With only 10 runs per repeat, this means a categorical
hparam will remain at the default or tuned value on most or even all runs, and so is
generally unperturbed within a repeat. I think this is why I am currently seeing only a
limited impact of hparam perturbation.


# Evaluation Procedure

Time permitting, I will examine each combination of model, classifier, and training perturbation schemes
in two scenarios:

## "Out-of-the-Box" Evaluations

Scikit-Learn, XGBoost, and PyTorch provide a number of default values for most hyperparameters.
Using these hparams is equivalent to using an "untuned" model, and we evaluate models using
defaults as a first step. In general, we should expect good PPSMs with default values or
perturbations thereof, because otherwise the defaults would not be good defaults that work
in a variety of situations.

**Note**: In fact, early testing to estimate runtimes used *completely random* hyperparameters, and
in most cases most models still performed reasonably well most of the time.


## Tuned Evaluations

I plan to use random search to tune each model-classifier-dataset combination
$4 \times 4 \times 37 = 592$ tunings. The hparams found by these tunings will then be
used for ALL subsequent repeats, runs, and combinations of perturbation scheme.

Arguably, we want to know how e.g. data perturbation and downsampling impact tuning too,
but then this gets inordinately expensive.

# Runtime Considerations

# Results



I've got all of the setup done and preliminary results for the model variance /
error consistency thing we discussed with Pascal Tyrrell.

There's a lot to describe, but the main thing is that models are fit subject to
a number of sources of variance (e.g. hyperparamter perturbation, data
perturbation, data downsampling) and then evaluated across *runs* and *repeats*.

Within a *repeat*, I do 10 fits / runs and get the predictions on a test set
X_test shared across these runs. Because X_test is shared, this allows
computing an EC, or any other Pairwise Prediction Similarity Metric (PPSM). I
also do many repeats (10) where this process is repeated, but with a different
shared X_test (chosen

This allows very rich summary of the PPSMs. For example, if our PPSM
of interest is the EC mean across the 45 unique error-set pairings, then we
get 10 such EC means, and so can talk about the distribution of this metric,
rather than just getting a single value for it. Because predictions are saved,
then
it is also trivial to compare the EC


====================================================================================
Accuracies

                              count   mean    std    min   2.5%    50%  97.5%    max
dataset_name classifier_kind
anneal       lr-sgd           100.0  0.941  0.015  0.920  0.920  0.938  0.982  0.982
             mlp              100.0  0.956  0.013  0.929  0.933  0.956  0.982  0.991
             svm-sgd          100.0  0.984  0.007  0.973  0.973  0.982  0.996  0.996
             xgb              100.0  0.996  0.006  0.982  0.982  0.998  1.000  1.000
vehicle      lr-sgd           100.0  0.919  0.010  0.901  0.901  0.920  0.939  0.939
             mlp              100.0  0.980  0.009  0.958  0.958  0.981  0.995  0.995
             svm-sgd          100.0  0.942  0.013  0.920  0.920  0.943  0.958  0.967
             xgb              100.0  0.986  0.010  0.962  0.962  0.988  1.000  1.000
====================================================================================
ECs (Error Set Intersection Divided by Test Size)

                              count   mean    std    min   2.5%    50%  97.5%    max
dataset_name classifier_kind
anneal       lr-sgd           450.0  0.057  0.016  0.018  0.018  0.062  0.076  0.080
             mlp              450.0  0.033  0.010  0.004  0.004  0.036  0.049  0.058
             svm-sgd          450.0  0.015  0.007  0.004  0.004  0.018  0.027  0.027
             xgb              450.0  0.004  0.006  0.000  0.000  0.002  0.018  0.018
vehicle      lr-sgd           450.0  0.081  0.010  0.061  0.061  0.080  0.094  0.099
             mlp              450.0  0.016  0.008  0.005  0.005  0.014  0.033  0.038
             svm-sgd          450.0  0.057  0.012  0.033  0.038  0.054  0.080  0.080
             xgb              450.0  0.014  0.010  0.000  0.000  0.012  0.038  0.038
====================================================================================
ECs (Error Set Intersection Divided by Error Set Union)

                              count   mean    std    min   2.5%    50%  97.5%  max
dataset_name classifier_kind
anneal       lr-sgd           450.0  0.919  0.103  0.500  0.571  0.933    1.0  1.0
             mlp              450.0  0.605  0.160  0.125  0.174  0.600    0.9  1.0
             svm-sgd          450.0  0.979  0.066  0.750  0.750  1.000    1.0  1.0
             xgb              450.0  0.500  0.501  0.000  0.000  0.500    1.0  1.0
vehicle      lr-sgd           450.0  0.993  0.018  0.938  0.938  1.000    1.0  1.0
             mlp              450.0  0.633  0.225  0.143  0.200  0.667    1.0  1.0
             svm-sgd          450.0  0.978  0.048  0.750  0.867  1.000    1.0  1.0
             xgb              450.0  0.900  0.300  0.000  0.000  1.000    1.0  1.0
====================================================================================

https://github.com/stfxecutables/model_variance/blob/master/PLANNING.md
