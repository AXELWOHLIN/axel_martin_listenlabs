import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq
from utils import nearest_psd_correlation

@dataclass
class CopulaModel:
    thresholds: np.ndarray        # t_k = Phi^{-1}(p_k)
    latent_corr: np.ndarray       # PSD correlation matrix for latent normals
    attr_ids: List[str]           # ordered list of constrained attribute ids

def _pair_p11_from_phi(p1: float, p2: float, phi: float) -> float:
    term = phi * np.sqrt(p1*(1-p1)*p2*(1-p2))
    p11 = p1*p2 + term
    # Feasible bounds
    lo = max(0.0, p1 + p2 - 1.0)
    hi = min(p1, p2)
    return float(np.clip(p11, lo, hi))

def _invert_tetrachoric(t1: float, t2: float, p11: float) -> float:
    """
    Solve for latent correlation rho such that
    BVN_CDF([t1,t2]; rho) == p11. Uses Brent on [-0.999, 0.999].
    """
    def f(rho):
        cov = np.array([[1.0, rho],[rho, 1.0]])
        return multivariate_normal.cdf([t1, t2], mean=[0,0], cov=cov) - p11
    # Root-find
    return brentq(f, -0.999, 0.999)

def build_copula_for_constrained(
    relative_frequencies: Dict[str, float],
    correlations: Dict[str, Dict[str, float]],
    constrained_attr_ids: List[str]
) -> CopulaModel:
    # Order the K constrained attributes
    attrs = list(constrained_attr_ids)
    K = len(attrs)
    p = np.array([relative_frequencies[a] for a in attrs], dtype=float)
    t = norm.ppf(p)

    # Build latent correlation via pairwise inversion
    R = np.eye(K)
    for i in range(K):
        for j in range(i+1, K):
            a_i, a_j = attrs[i], attrs[j]
            phi = correlations.get(a_i, {}).get(a_j, correlations.get(a_j, {}).get(a_i, 0.0))
            p11 = _pair_p11_from_phi(p[i], p[j], phi)
            rho_ij = _invert_tetrachoric(t[i], t[j], p11)
            R[i, j] = R[j, i] = rho_ij

    R = nearest_psd_correlation(R)
    return CopulaModel(thresholds=t, latent_corr=R, attr_ids=attrs)

def sample_types(model: CopulaModel, n_samples: int, rng: np.random.Generator):
    """
    Sample n_samples of the K constrained attributes.
    Returns:
      types_bit: (U, K) unique patterns, weights: (U,) probabilities
    """
    K = len(model.attr_ids)
    Z = rng.multivariate_normal(mean=np.zeros(K), cov=model.latent_corr, size=n_samples)
    A = (Z <= model.thresholds).astype(np.uint8)

    # unique types and weights
    # encode rows as bitstrings for grouping
    bits = (A * (1 << np.arange(K))).sum(axis=1)
    uniq, counts = np.unique(bits, return_counts=True)
    weights = counts / counts.sum()

    # decode bits back to 0/1 arrays
    U = len(uniq)
    types = np.zeros((U, K), dtype=np.uint8)
    for r, b in enumerate(uniq):
        for k in range(K):
            types[r, k] = (b >> k) & 1
    return types, weights
