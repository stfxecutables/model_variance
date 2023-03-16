from __future__ import annotations

# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ndarray
from numpy.random import Generator


def sig_perturb(x: ndarray, n_digits: int = 1) -> ndarray:
    delta = 10 ** (np.floor(np.log10(np.abs(x))) / 10**n_digits)
    return x + delta * np.random.uniform(-1, 1, x.shape)


def sig_perturb_plus(
    x: ndarray, n_digits: int = 1, rng: Optional[Generator] = None
) -> ndarray:
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x)
    idx = x != 0
    delta = 10 ** (np.floor(np.log10(np.abs(x[idx]))) / 10**n_digits)
    # delta[idx] = 0
    if n_digits == 1:
        delta *= 2
    perturbed = deepcopy(x)
    perturbed[idx] += delta * rng.uniform(-1, 1, [len(delta)])
    return perturbed


def percent_perturb(x: float, xmin: float, xmax: float, magnitude: float) -> ndarray:
    """Perturb to within `magnitude` percent of the range of `x`"""
    raise NotImplementedError()


def neighbour_perturb(
    x: ndarray, distances: ndarray, scale: float, rng: Optional[Generator] = None
) -> ndarray:
    """
    We use the "normalized Gaussians" method to get a random perturbation
    vector for each sample in d_nn / 2, where d_nn = sqrt(|x^2 - x_nn^2|) for
    sample x and nearest neighor x_nn
    """
    rng = np.random.default_rng() if rng is None else rng
    deltas = distances / scale
    gaussians = rng.standard_normal(x.shape)
    norm = np.linalg.norm(gaussians, axis=1, keepdims=True)
    shell = gaussians / norm  # samples on n-sphere shell
    radii = rng.uniform(0, 1, size=x.shape) * deltas[:, None]
    perturbations = radii * shell
    perturbed = x + perturbations
    assert np.all(np.linalg.norm(perturbed - x, axis=1) <= deltas)
    return perturbed
