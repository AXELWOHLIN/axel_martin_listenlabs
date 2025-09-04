from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple
from planner import solve_acceptance_lp, LPPlan

@dataclass
class PolicyState:
    admitted: int = 0
    rejected: int = 0
    counts: np.ndarray = None  # (K,) admitted counts per constrained attr
    last_plan: LPPlan = None

def hoeffding_safety(N: int, K: int, delta_total: float = 0.01) -> float:
    """
    Uniform epsilon for all constraints using union bound.
    """
    delta_k = max(delta_total / max(K, 1), 1e-6)
    eps = np.sqrt(np.log(2.0 / delta_k) / (2.0 * N))
    return float(eps)

def build_plan_for_residual(types, weights, m_remaining, N_remaining, eps):
    r_targets = np.minimum(1.0, (m_remaining / max(N_remaining, 1e-9)) + eps)
    return solve_acceptance_lp(types, weights, r_targets)

class AcceptancePolicy:
    def __init__(self, types, weights, K, m_targets, N_total=1000, seed=0, block=50):
        self.types = types
        self.weights = weights
        self.K = K
        self.N_total = N_total
        self.block = block
        self.rng = np.random.default_rng(seed)
        self.state = PolicyState(admitted=0, rejected=0, counts=np.zeros(K, dtype=int))
        self.m_targets = np.array(m_targets, dtype=int)
        self.eps = hoeffding_safety(N_total, K, 0.01)
        # initial plan on full residuals
        plan = build_plan_for_residual(self.types, self.weights, self.m_targets, N_total, self.eps)
        self.state.last_plan = plan

    def _bit_from_attrs(self, a_vec: np.ndarray) -> int:
        # a_vec is 0/1 array of length K in our attribute order
        return int((a_vec * (1 << np.arange(self.K))).sum())

    def _alpha_for(self, a_vec: np.ndarray) -> float:
        # locate type row (use bit-key hash)
        bit = self._bit_from_attrs(a_vec)
        # decode all types to bits once
        if not hasattr(self, "_type_bits"):
            self._type_bits = (self.types * (1 << np.arange(self.K))).sum(axis=1)
        idx = np.where(self._type_bits == bit)[0]
        if len(idx) == 0:
            return 0.0  # unseen combination -> conservative
        return float(self.state.last_plan.alpha[idx[0]])

    def decide(self, a_vec: np.ndarray) -> bool:
        # hard feasibility guard for any extremely tight single-constraint
        N_rem = self.N_total - self.state.admitted
        if N_rem <= 0:
            return False
        for k in range(self.K):
            need_k = max(0, self.m_targets[k] - self.state.counts[k])
            if need_k > N_rem:
                # every remaining seat must cover attr k
                if a_vec[k] != 1:
                    self.state.rejected += 1
                    return False

        # score via alpha
        alpha = self._alpha_for(a_vec)
        accept = self.rng.random() < alpha
        if accept:
            self.state.admitted += 1
            self.state.counts += a_vec.astype(int)
        else:
            self.state.rejected += 1

        # re-solve periodically or if we are behind
        if self.state.admitted % self.block == 0 or accept is False:
            m_rem = np.maximum(0, self.m_targets - self.state.counts)
            N_rem2 = self.N_total - self.state.admitted
            self.state.last_plan = build_plan_for_residual(self.types, self.weights, m_rem, N_rem2, self.eps)
        return accept
