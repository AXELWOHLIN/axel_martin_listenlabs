import numpy as np
from typing import Tuple
from dataclasses import dataclass
from scipy.optimize import linprog

@dataclass
class LPPlan:
    types: np.ndarray        # (U, K) 0/1
    weights: np.ndarray      # (U,)
    alpha: np.ndarray        # (U,) acceptance prob per type in [0,1]
    r_targets: np.ndarray    # (K,) tightened shares used in this solve
    duals: np.ndarray        # (K,) shadow prices (if available)

def solve_acceptance_lp(types: np.ndarray, weights: np.ndarray, r_targets: np.ndarray) -> LPPlan:
    """
    Solve:
      max sum_u w_u * alpha_u
      s.t. sum_u w_u * alpha_u * (types[u,k] - r_k) >= 0,  for k=1..K
           0 <= alpha_u <= 1
    Implemented as a minimization in linprog by negating the objective.
    """
    U, K = types.shape
    c = -weights.copy()  # minimize -p => maximize p
    # A_ub x <= b_ub  -> we have A * alpha >= 0 => -A * alpha <= 0
    A = np.zeros((K, U))
    for k in range(K):
        A[k, :] = -weights * (types[:, k] - r_targets[k])
    b = np.zeros(K)

    bounds = [(0.0, 1.0)] * U
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        # Infeasible under current targets; caller should relax or increase safety.
        alpha = np.clip(np.zeros(U), 0, 1)
        duals = np.zeros(K)
    else:
        alpha = np.clip(res.x, 0.0, 1.0)
        # HiGHS returns reduced costs/dual info; may vary by SciPy version
        # If available:
        duals = getattr(res, "ineqlin", None)
        if isinstance(duals, dict) and "marginals" in duals:
            duals = np.array(duals["marginals"])
        else:
            duals = np.zeros(K)

    return LPPlan(types=types, weights=weights, alpha=alpha, r_targets=r_targets, duals=duals)
