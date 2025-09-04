0) TL;DR (How this wins)

Goal equivalence. Minimizing rejections until you admit 
𝑁
N is equivalent to maximizing the unconditional acceptance probability 
𝑝
p. If your policy accepts a random arrival with prob. 
𝑝
p, the expected number of rejections before hitting 
𝑁
N admits is 
𝑁
1
−
𝑝
𝑝
N
p
1−p
	​

.

Optimal stationary acceptance region (fluid model). Given the true i.i.d. joint distribution 
𝜋
(
𝑎
)
π(a) of constrained attributes 
𝑎
∈
{
0
,
1
}
𝐾
a∈{0,1}
K
, the best stationary policy solves a small linear program (LP) that picks which types to accept (possibly with randomization) to maximize 
𝑝
p while ensuring each required share is met in the accepted cohort. This is the “deterministic LP / fluid approximation” widely used in revenue management and online allocation; we then re-solve it as we go so realized randomness doesn’t derail us. 
JSTOR
arXiv

We don’t know 
𝜋
(
𝑎
)
π(a) exactly; the API gives marginals and pairwise correlations. We build a Gaussian‑copula (multivariate probit) surrogate that matches the given marginals and pairwise associations, then Monte‑Carlo sample it to estimate 
𝜋
(
𝑎
)
π(a) over the constrained attributes only. This is the Emrich–Piedmonte approach and variants that are standard for correlated binary simulation. 
JSTOR
R Project Search
SAS Blogs

Numerics: We map binary pairwise correlations to latent normal correlations by solving bivariate normal CDF equations; we then project the matrix to the nearest PSD correlation matrix (Higham). All in SciPy/NumPy. 
SciPy Documentation
Oxford Academic
eprints.maths.manchester.ac.uk

Online control: We (i) re-solve the LP every few dozen admits (“re-solving heuristic”) and (ii) add Hoeffding‑based safety buffers to the required shares to get high‑probability feasibility at 
𝑁
N. 
JSTOR
arXiv
Wikipedia




5) Code skeletons (ready to drop into Cursor)

These are minimal but complete building blocks. You can paste them into the files listed in §1 and implement the TODOs.

utils.py
import numpy as np

def nearest_psd_correlation(C: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest correlation matrix by
    eigenvalue clipping and re-scaling the diagonal to ones.
    Higham (2002) idea; good enough for our use.  # noqa
    """
    # symmetrize
    C = 0.5 * (C + C.T)
    # eigen clip
    w, V = np.linalg.eigh(C)
    w_clipped = np.clip(w, tol, None)
    C_psd = (V * w_clipped) @ V.T
    # set diag to 1
    d = np.sqrt(np.diag(C_psd))
    C_corr = C_psd / np.outer(d, d)
    np.fill_diagonal(C_corr, 1.0)
    return C_corr


Source: Higham’s nearest correlation matrix. 
Oxford Academic
eprints.maths.manchester.ac.uk

model.py (Gaussian‑copula for constrained attrs only)
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq
from .utils import nearest_psd_correlation

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


Docs: Multivariate normal and CDF; phi/tetrachoric background. 
SciPy Documentation
+1
Wikipedia
Stata

planner.py (LP to maximize acceptance rate)
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


SciPy’s HiGHS LP routines (objective sign, bounds, and access to duals). 
SciPy Documentation
+1

policy.py (controller + safety + re-solving)
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple
from .planner import solve_acceptance_lp, LPPlan

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


Safety buffer from Hoeffding. 
Wikipedia

Re-solving every block admits (and on “misses”) keeps us on track. 
JSTOR
arXiv

client.py (HTTP wrapper for your API)
import requests
from typing import Dict, Any

class GameClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def new_game(self, scenario: int, player_id: str) -> Dict[str, Any]:
        url = f"{self.base}/new-game"
        params = {"scenario": scenario, "playerId": player_id}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def decide_and_next(self, game_id: str, person_index: int, accept: bool | None) -> Dict[str, Any]:
        url = f"{self.base}/decide-and-next"
        params = {"gameId": game_id, "personIndex": person_index}
        if person_index > 0 or accept is not None:
            params["accept"] = "true" if accept else "false"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

runner.py (main loop glue)
import numpy as np
from .client import GameClient
from .model import build_copula_for_constrained, sample_types
from .policy import AcceptancePolicy

def run_game(base_url: str, scenario: int, player_id: str, seed: int = 42, samples: int = 200_000):
    gc = GameClient(base_url)
    meta = gc.new_game(scenario, player_id)

    game_id = meta["gameId"]
    constraints = meta["constraints"]
    stats = meta["attributeStatistics"]

    # Keep only the attributes that appear in constraints
    attr_ids = [c["attribute"] for c in constraints]
    m_targets = [c["minCount"] for c in constraints]

    # Build Gaussian-copula model for those attributes only
    model = build_copula_for_constrained(
        relative_frequencies=stats["relativeFrequencies"],
        correlations=stats["correlations"],
        constrained_attr_ids=attr_ids
    )

    # Monte Carlo estimate of type distribution for constrained attrs
    rng = np.random.default_rng(seed)
    types, weights = sample_types(model, n_samples=samples, rng=rng)

    # Index map to retrieve order
    attr_index = {a: i for i, a in enumerate(model.attr_ids)}

    # Start streaming people
    policy = AcceptancePolicy(types, weights, len(attr_ids), m_targets, N_total=1000, seed=seed, block=50)

    # first call (personIndex=0) to get the first person
    resp = gc.decide_and_next(game_id, person_index=0, accept=None)

    while resp["status"] == "running":
        person = resp["nextPerson"]
        a_vec = np.zeros(len(attr_ids), dtype=np.uint8)
        # Extract this person's constrained attrs in our order
        for a_id, val in person["attributes"].items():
            if a_id in attr_index:
                a_vec[attr_index[a_id]] = 1 if val else 0

        decision = policy.decide(a_vec)
        resp = gc.decide_and_next(
            game_id=game_id,
            person_index=person["personIndex"],
            accept=bool(decision)
        )

    # Completed or failed
    return resp