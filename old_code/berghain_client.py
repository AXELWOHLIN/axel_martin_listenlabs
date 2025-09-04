#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib import parse, request


CAPACITY = 1000


# --- helper math for scenario 2 ---
def _clip01(x: float, eps: float = 1e-9) -> float:
    return min(1 - eps, max(eps, x))


def _logit(p: float) -> float:
    p = _clip01(p)
    return math.log(p / (1 - p))


@dataclass
class Constraint:
    attribute: str
    min_count: int


class SimpleLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        # truncate at start
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")

    def write(self, record: Dict):
        record_with_ts = dict(record)
        record_with_ts["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_with_ts, separators=(",", ":")) + "\n")


class BerghainClient:
    def __init__(self, base_url: str, player_id: str, scenario: int, log_path: str):
        self.base_url = base_url.rstrip("/")
        self.player_id = player_id
        self.scenario = scenario
        self.logger = SimpleLogger(log_path)

        self.game_id: Optional[str] = None
        self.constraints: Dict[str, Constraint] = {}
        self.required_counts: Dict[str, int] = {}
        self.attr_accept_counts: Dict[str, int] = {}
        self.attribute_statistics = {}

        self.admitted_count = 0
        self.rejected_count = 0

    def _http_get(self, path: str, params: Dict[str, str]) -> Dict:
        url = f"{self.base_url}{path}?{parse.urlencode(params)}"
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=60) as resp:
            body = resp.read()
            return json.loads(body.decode("utf-8"))

    def _infer_required(self, min_count_value) -> int:
        # Supports either absolute counts or fractions in [0,1].
        try:
            v = float(min_count_value)
        except Exception:
            return int(min_count_value)
        if 0.0 <= v <= 1.0:
            return int(math.ceil(v * CAPACITY))
        return int(round(v))

    def start_new_game(self):
        data = self._http_get(
            "/new-game",
            {
                "scenario": str(self.scenario),
                "playerId": self.player_id,
            },
        )
        self.game_id = data["gameId"]

        constraints_list = data.get("constraints", [])
        self.constraints = {}
        self.required_counts = {}
        self.attr_accept_counts = {}
        for c in constraints_list:
            attr = c["attribute"]
            required = self._infer_required(c["minCount"])
            self.constraints[attr] = Constraint(attribute=attr, min_count=required)
            self.required_counts[attr] = required
            self.attr_accept_counts[attr] = 0

        self.attribute_statistics = data.get("attributeStatistics", {})

        strategy_name = "strategy1"
        if self.scenario == 2:
            strategy_name = "strategy2"
        elif self.scenario == 3:
            strategy_name = "strategy3"

        self.logger.write(
            {
                "event": "start",
                "scenario": self.scenario,
                "strategy": strategy_name,
                "gameId": self.game_id,
                "capacity": CAPACITY,
                "constraints": self.required_counts,
                "attributeStatistics": self.attribute_statistics,
            }
        )

        # Initialize intersection counters for strategies that use them
        if not hasattr(self, "intersection_accept_counts"):
            self.intersection_accept_counts = {}

    def decide_and_fetch(self, person_index: int, accept: Optional[bool]) -> Dict:
        params = {
            "gameId": self.game_id,
            "personIndex": str(person_index),
        }
        if accept is not None:
            params["accept"] = "true" if accept else "false"
        return self._http_get("/decide-and-next", params)

    # Strategy 1 (default):
    # - If all constraints satisfied, accept everyone until full
    # - Otherwise, accept person if they satisfy any still-needed constrained attribute
    # - If not helpful, accept only if doing so keeps meeting constraints feasible
    def strategy1_should_accept(self, person_attrs: Dict[str, bool]) -> Tuple[bool, Dict[str, object]]:
        all_satisfied = self._all_constraints_satisfied()
        remaining_slots = CAPACITY - self.admitted_count
        deficit = self._total_deficit()

        helpful = False
        for attr, required in self.required_counts.items():
            if self.attr_accept_counts.get(attr, 0) < required and person_attrs.get(attr, False):
                helpful = True
                break

        if all_satisfied:
            return True, {
                "allSatisfiedBefore": True,
                "helpful": helpful,
                "remainingSlotsBefore": remaining_slots,
                "deficitBefore": deficit,
            }

        if helpful:
            return True, {
                "allSatisfiedBefore": False,
                "helpful": True,
                "remainingSlotsBefore": remaining_slots,
                "deficitBefore": deficit,
            }

        # Accept only if we can still meet the remaining deficits after this acceptance
        safe_to_accept = (remaining_slots - 1) >= deficit
        return safe_to_accept, {
            "allSatisfiedBefore": False,
            "helpful": False,
            "remainingSlotsBefore": remaining_slots,
            "deficitBefore": deficit,
        }

    # Strategy 2 (scaffold): customize acceptance logic for scenario 2 here.
    # Implements guards → guardrail → KL scoring policy
    def strategy2_should_accept(self, person_attrs: Dict[str, bool]) -> Tuple[bool, Dict[str, object]]:
        remaining_slots = CAPACITY - self.admitted_count
        if remaining_slots <= 0:
            return False, {"reason": "capacity_full", "remainingSlotsBefore": remaining_slots}

        # Attribute keys
        T_key = "techno_lover"
        W_key = "well_connected"
        C_key = "creative"
        L_key = "berlin_local"

        # Current counts
        T = self._get_attr_count(T_key)
        W = self._get_attr_count(W_key)
        C = self._get_attr_count(C_key)
        L = self._get_attr_count(L_key)
        LT = self._get_intersection_count(T_key, L_key)
        CL = self._get_intersection_count(C_key, L_key)
        n = self.admitted_count  # alias if needed
        s = remaining_slots

        # Requirements
        r_T = int(self.required_counts.get(T_key, 0))
        r_W = int(self.required_counts.get(W_key, 0))
        r_C = int(self.required_counts.get(C_key, 0))
        r_L = int(self.required_counts.get(L_key, 0))

        # Derived intersection requirement
        min_LT_needed = max(0, r_T - (CAPACITY - r_L))
        min_CL_needed = max(0, r_C - (CAPACITY - r_L))

        # Step A: Feasibility guards
        guards_violated = []
        if T < r_T - s:
            guards_violated.append(T_key)
        if W < r_W - s:
            guards_violated.append(W_key)
        if C < r_C - s:
            guards_violated.append(C_key)
        if L < r_L - s:
            guards_violated.append(L_key)

        lt_guard_violated = LT < (min_LT_needed - s)
        cl_guard_violated = CL < (min_CL_needed - s)

        helps_single = any(person_attrs.get(q, False) for q in guards_violated)
        helps_intersection = person_attrs.get(T_key, False) and person_attrs.get(L_key, False)
        helps_intersection_cl = person_attrs.get(C_key, False) and person_attrs.get(L_key, False)

        if guards_violated or lt_guard_violated or cl_guard_violated:
            if helps_single or (lt_guard_violated and helps_intersection) or (cl_guard_violated and helps_intersection_cl):
                return True, {
                    "stage": "guards",
                    "guardsViolated": guards_violated,
                    "ltGuardViolated": lt_guard_violated,
                    "clGuardViolated": cl_guard_violated,
                    "remainingSlotsBefore": s,
                }
            else:
                return False, {
                    "stage": "guards",
                    "guardsViolated": guards_violated,
                    "ltGuardViolated": lt_guard_violated,
                    "clGuardViolated": cl_guard_violated,
                    "remainingSlotsBefore": s,
                }

        # Scarcity gating using urgency before guardrails
        def need_frac(needed: int, have: int, slots: int) -> float:
            return _clip01(max(0, needed - have) / slots) if slots > 0 else 0.0

        u_T = need_frac(r_T, T, s)
        u_W = need_frac(r_W, W, s)
        u_C = need_frac(r_C, C, s)
        u_L = need_frac(r_L, L, s)
        u_LT = _clip01(max(0, min_LT_needed - LT) / s) if s > 0 else 0.0
        u_CL = _clip01(max(0, min_CL_needed - CL) / s) if s > 0 else 0.0

        if (
            u_C >= 0.25
            and (not person_attrs.get(C_key, False))
            and (not helps_intersection)
            and (not helps_single)
            and (not person_attrs.get(L_key, False))  # allow locals through even if not creative
        ):
            return False, {
                "stage": "scarcity_gate",
                "reason": "need_creatives",
                "u_C": u_C,
                "remainingSlotsBefore": s,
            }

        if u_L >= 0.60 and (not person_attrs.get(L_key, False)) and (not helps_intersection) and (not helps_single):
            return False, {
                "stage": "scarcity_gate",
                "reason": "need_locals",
                "u_L": u_L,
                "remainingSlotsBefore": s,
            }

        # Step B: Correlation guardrail (avoid too many local non-tech)
        will_be_L = L + (1 if person_attrs.get(L_key, False) else 0)
        will_be_LT = LT + (1 if (person_attrs.get(L_key, False) and person_attrs.get(T_key, False)) else 0)
        guardrail_threshold = will_be_L - (CAPACITY - r_T)

        if person_attrs.get(L_key, False) and not person_attrs.get(T_key, False):
            if will_be_LT < guardrail_threshold:
                return False, {
                    "stage": "guardrail",
                    "reason": "local_non_tech_would_break_LT_ge_L_minus_(N-rT)",
                    "guardrailThresholdAfter": guardrail_threshold,
                    "LT_after": will_be_LT,
                    "L_after": will_be_L,
                }

        # Symmetric: avoid too many non-local techno (to keep LT feasible)
        will_be_T = T + (1 if person_attrs.get(T_key, False) else 0)
        guardrail_threshold_T = will_be_T - (CAPACITY - r_L)
        if person_attrs.get(T_key, False) and not person_attrs.get(L_key, False):
            if will_be_LT < guardrail_threshold_T:
                return False, {
                    "stage": "guardrail",
                    "reason": "non_local_tech_would_break_LT_ge_T_minus_(N-rL)",
                    "guardrailThresholdAfter": guardrail_threshold_T,
                    "LT_after": will_be_LT,
                    "T_after": will_be_T,
                }

        # Additional guardrail for CL: avoid too many local non-creative
        will_be_CL = CL + (1 if (person_attrs.get(L_key, False) and person_attrs.get(C_key, False)) else 0)
        guardrail_threshold_cl = will_be_L - (CAPACITY - r_C)
        if person_attrs.get(L_key, False) and not person_attrs.get(C_key, False):
            if will_be_CL < guardrail_threshold_cl:
                return False, {
                    "stage": "guardrail",
                    "reason": "local_non_creative_would_break_CL_ge_L_minus_(N-rC)",
                    "guardrailThresholdAfter": guardrail_threshold_cl,
                    "CL_after": will_be_CL,
                    "L_after": will_be_L,
                }

        # Post-accept feasibility: do not accept if it would make any quota infeasible
        infeasible_reasons = self._would_break_feasibility_after_accept(
            person_attrs,
            required={T_key: r_T, W_key: r_W, C_key: r_C, L_key: r_L},
            current={T_key: T, W_key: W, C_key: C, L_key: L},
            s=remaining_slots,
            min_LT_needed=min_LT_needed,
            LT_current=LT,
        )
        if infeasible_reasons:
            return False, {
                "stage": "post_feasibility",
                "reasons": infeasible_reasons,
                "remainingSlotsBefore": remaining_slots,
            }

        # Step C: KL-optimal scoring when feasible
        params = self._scenario2_pop_params()
        p = params["p"]
        p_LT = params["p_LT"]
        p_CL = params["p_CL"]

        lam_T = _logit(u_T) - _logit(p[T_key])
        lam_W = _logit(u_W) - _logit(p[W_key])
        lam_C = _logit(u_C) - _logit(p[C_key])
        lam_L = _logit(u_L) - _logit(p[L_key])
        lam_LT = _logit(u_LT) - _logit(p_LT)
        lam_CL = _logit(u_CL) - _logit(p_CL)

        x_T = 1 if person_attrs.get(T_key, False) else 0
        x_W = 1 if person_attrs.get(W_key, False) else 0
        x_C = 1 if person_attrs.get(C_key, False) else 0
        x_L = 1 if person_attrs.get(L_key, False) else 0
        x_LT = x_L * x_T
        x_CL = x_L * (1 if person_attrs.get(C_key, False) else 0)

        S = (lam_T * x_T) + (lam_W * x_W) + (lam_C * x_C) + (lam_L * x_L) + (lam_LT * x_LT) + (lam_CL * x_CL)

        accept = S >= 0.0
        return accept, {
            "stage": "scoring",
            "score": S,
            "lambdas": {"T": lam_T, "W": lam_W, "C": lam_C, "L": lam_L, "LT": lam_LT, "CL": lam_CL},
            "urgencies": {"u_T": u_T, "u_W": u_W, "u_C": u_C, "u_L": u_L, "u_LT": u_LT, "u_CL": u_CL},
            "remainingSlotsBefore": s,
        }

    # Strategy 3: guards → guardrails (German-International) → KL scoring with GI and QV intersections
    def strategy3_should_accept(self, person_attrs: Dict[str, bool]) -> Tuple[bool, Dict[str, object]]:
        remaining_slots = CAPACITY - self.admitted_count
        if remaining_slots <= 0:
            return False, {"reason": "capacity_full", "remainingSlotsBefore": remaining_slots}

        # Attribute keys (scenario 3)
        G_key = "german_speaker"
        I_key = "international"
        U_key = "underground_veteran"
        F_key = "fashion_forward"
        Q_key = "queer_friendly"
        V_key = "vinyl_collector"

        # Current counts
        G = self._get_attr_count(G_key)
        I = self._get_attr_count(I_key)
        U = self._get_attr_count(U_key)
        F = self._get_attr_count(F_key)
        Q = self._get_attr_count(Q_key)
        V = self._get_attr_count(V_key)
        GI = self._get_intersection_count(G_key, I_key)
        QV = self._get_intersection_count(Q_key, V_key)
        s = remaining_slots

        # Requirements
        r_G = int(self.required_counts.get(G_key, 0))
        r_I = int(self.required_counts.get(I_key, 0))
        r_U = int(self.required_counts.get(U_key, 0))
        r_F = int(self.required_counts.get(F_key, 0))
        r_Q = int(self.required_counts.get(Q_key, 0))
        r_V = int(self.required_counts.get(V_key, 0))

        # Derived intersections
        min_GI_needed = max(0, r_G - (CAPACITY - r_I))
        min_QV_needed = max(0, r_Q - (CAPACITY - r_V))

        # Guards
        guards_violated = []
        if G < r_G - s: guards_violated.append(G_key)
        if I < r_I - s: guards_violated.append(I_key)
        if U < r_U - s: guards_violated.append(U_key)
        if F < r_F - s: guards_violated.append(F_key)
        if Q < r_Q - s: guards_violated.append(Q_key)
        if V < r_V - s: guards_violated.append(V_key)
        gi_guard_violated = GI < (min_GI_needed - s)
        qv_guard_violated = QV < (min_QV_needed - s)

        helps_single = any(person_attrs.get(k, False) for k in guards_violated)
        helps_gi = person_attrs.get(G_key, False) and person_attrs.get(I_key, False)
        helps_qv = person_attrs.get(Q_key, False) and person_attrs.get(V_key, False)

        if guards_violated or gi_guard_violated or qv_guard_violated:
            if helps_single or (gi_guard_violated and helps_gi) or (qv_guard_violated and helps_qv):
                return True, {"stage": "guards", "guardsViolated": guards_violated, "giGuardViolated": gi_guard_violated, "qvGuardViolated": qv_guard_violated, "remainingSlotsBefore": s}
            else:
                return False, {"stage": "guards", "guardsViolated": guards_violated, "giGuardViolated": gi_guard_violated, "qvGuardViolated": qv_guard_violated, "remainingSlotsBefore": s}

        # Scarcity gating for rare quotas: queer_friendly and vinyl_collector
        def need_frac(needed: int, have: int, slots: int) -> float:
            return _clip01(max(0, needed - have) / slots) if slots > 0 else 0.0

        u_G = need_frac(r_G, G, s)
        u_I = need_frac(r_I, I, s)
        u_U = need_frac(r_U, U, s)
        u_F = need_frac(r_F, F, s)
        u_Q = need_frac(r_Q, Q, s)
        u_V = need_frac(r_V, V, s)
        u_GI = _clip01(max(0, min_GI_needed - GI) / s) if s > 0 else 0.0
        u_QV = _clip01(max(0, min_QV_needed - QV) / s) if s > 0 else 0.0

        # when very behind on Q or V, prefer profiles that help those or GI
        if u_Q >= 0.20 and (not person_attrs.get(Q_key, False)) and (not helps_gi) and (not helps_single):
            return False, {"stage": "scarcity_gate", "reason": "need_queer_friendly", "u_Q": u_Q, "remainingSlotsBefore": s}
        if u_V >= 0.20 and (not person_attrs.get(V_key, False)) and (not helps_gi) and (not helps_single):
            return False, {"stage": "scarcity_gate", "reason": "need_vinyl_collectors", "u_V": u_V, "remainingSlotsBefore": s}

        # Guardrails for GI: avoid too many international non-German and German non-international when GI would fail
        will_be_G = G + (1 if person_attrs.get(G_key, False) else 0)
        will_be_I = I + (1 if person_attrs.get(I_key, False) else 0)
        will_be_GI = GI + (1 if (person_attrs.get(G_key, False) and person_attrs.get(I_key, False)) else 0)
        thr_from_I = will_be_I - (CAPACITY - r_G)
        thr_from_G = will_be_G - (CAPACITY - r_I)
        if person_attrs.get(I_key, False) and not person_attrs.get(G_key, False):
            if will_be_GI < thr_from_I:
                return False, {"stage": "guardrail", "reason": "intl_non_german_would_break_GI", "GI_after": will_be_GI, "threshold": thr_from_I}
        if person_attrs.get(G_key, False) and not person_attrs.get(I_key, False):
            if will_be_GI < thr_from_G:
                return False, {"stage": "guardrail", "reason": "german_non_intl_would_break_GI", "GI_after": will_be_GI, "threshold": thr_from_G}

        # Post-accept feasibility considering singles and intersections GI and QV
        infeasible = self._s3_would_break_feasibility_after_accept(
            person_attrs,
            required={G_key: r_G, I_key: r_I, U_key: r_U, F_key: r_F, Q_key: r_Q, V_key: r_V},
            current={G_key: G, I_key: I, U_key: U, F_key: F, Q_key: Q, V_key: V},
            s=s,
            min_GI=min_GI_needed,
            min_QV=min_QV_needed,
            GI_current=GI,
            QV_current=QV,
        )
        if infeasible:
            return False, {"stage": "post_feasibility", "reasons": infeasible, "remainingSlotsBefore": s}

        # Scoring with KL multipliers
        params = self._s3_pop_params()
        p = params["p"]
        p_GI = params["p_GI"]
        p_QV = params["p_QV"]

        lam_G = _logit(u_G) - _logit(p[G_key])
        lam_I = _logit(u_I) - _logit(p[I_key])
        lam_U = _logit(u_U) - _logit(p[U_key])
        lam_F = _logit(u_F) - _logit(p[F_key])
        lam_Q = _logit(u_Q) - _logit(p[Q_key])
        lam_V = _logit(u_V) - _logit(p[V_key])
        lam_GI = _logit(u_GI) - _logit(p_GI)
        lam_QV = _logit(u_QV) - _logit(p_QV)

        x_G = 1 if person_attrs.get(G_key, False) else 0
        x_I = 1 if person_attrs.get(I_key, False) else 0
        x_U = 1 if person_attrs.get(U_key, False) else 0
        x_F = 1 if person_attrs.get(F_key, False) else 0
        x_Q = 1 if person_attrs.get(Q_key, False) else 0
        x_V = 1 if person_attrs.get(V_key, False) else 0
        x_GI = x_G * x_I
        x_QV = x_Q * x_V

        S = (
            lam_G * x_G
            + lam_I * x_I
            + lam_U * x_U
            + lam_F * x_F
            + lam_Q * x_Q
            + lam_V * x_V
            + lam_GI * x_GI
            + lam_QV * x_QV
        )

        accept = S >= 0.0
        return accept, {
            "stage": "scoring",
            "score": S,
            "lambdas": {"G": lam_G, "I": lam_I, "U": lam_U, "F": lam_F, "Q": lam_Q, "V": lam_V, "GI": lam_GI, "QV": lam_QV},
            "urgencies": {"u_G": u_G, "u_I": u_I, "u_U": u_U, "u_F": u_F, "u_Q": u_Q, "u_V": u_V, "u_GI": u_GI, "u_QV": u_QV},
            "remainingSlotsBefore": s,
        }

    def _s3_pop_params(self):
        if hasattr(self, "_scenario3_cache"):
            return self._scenario3_cache
        rel = (self.attribute_statistics or {}).get("relativeFrequencies", {})
        corr = (self.attribute_statistics or {}).get("correlations", {})

        p_G = float(rel.get("german_speaker", 0.0))
        p_I = float(rel.get("international", 0.0))
        p_U = float(rel.get("underground_veteran", 0.0))
        p_F = float(rel.get("fashion_forward", 0.0))
        p_Q = float(rel.get("queer_friendly", 0.0))
        p_V = float(rel.get("vinyl_collector", 0.0))

        rho_GI = float(((corr.get("german_speaker", {}) or {}).get("international", 0.0)))
        if rho_GI == 0.0 and "german_speaker" in corr and "international" in corr["international"]:
            rho_GI = float(corr["international"]["german_speaker"])
        rho_QV = float(((corr.get("queer_friendly", {}) or {}).get("vinyl_collector", 0.0)))
        if rho_QV == 0.0 and "vinyl_collector" in corr and "queer_friendly" in corr["vinyl_collector"]:
            rho_QV = float(corr["vinyl_collector"]["queer_friendly"])

        var_G = p_G * (1 - p_G)
        var_I = p_I * (1 - p_I)
        var_Q = p_Q * (1 - p_Q)
        var_V = p_V * (1 - p_V)

        p_GI = p_G * p_I + rho_GI * math.sqrt(max(0.0, var_G * var_I))
        p_QV = p_Q * p_V + rho_QV * math.sqrt(max(0.0, var_Q * var_V))
        p_GI = _clip01(p_GI)
        p_QV = _clip01(p_QV)

        self._scenario3_cache = {
            "p": {
                "german_speaker": p_G,
                "international": p_I,
                "underground_veteran": p_U,
                "fashion_forward": p_F,
                "queer_friendly": p_Q,
                "vinyl_collector": p_V,
            },
            "p_GI": p_GI,
            "p_QV": p_QV,
        }
        return self._scenario3_cache

    def _s3_would_break_feasibility_after_accept(
        self,
        person_attrs: Dict[str, bool],
        required: Dict[str, int],
        current: Dict[str, int],
        s: int,
        min_GI: int,
        min_QV: int,
        GI_current: int,
        QV_current: int,
    ) -> Dict[str, object]:
        breaks = {}
        for key, needed in required.items():
            have = int(current.get(key, 0))
            delta = 1 if person_attrs.get(key, False) else 0
            if have + delta + max(0, s - 1) < int(needed):
                breaks[key] = {"have": have, "delta": delta, "slotsAfter": max(0, s - 1), "needed": int(needed)}
        delta_GI = 1 if (person_attrs.get("german_speaker") and person_attrs.get("international")) else 0
        delta_QV = 1 if (person_attrs.get("queer_friendly") and person_attrs.get("vinyl_collector")) else 0
        if GI_current + delta_GI + max(0, s - 1) < int(min_GI):
            breaks["GI"] = {"have": GI_current, "delta": delta_GI, "slotsAfter": max(0, s - 1), "needed": int(min_GI)}
        if QV_current + delta_QV + max(0, s - 1) < int(min_QV):
            breaks["QV"] = {"have": QV_current, "delta": delta_QV, "slotsAfter": max(0, s - 1), "needed": int(min_QV)}
        return breaks

    def _would_break_feasibility_after_accept(
        self,
        person_attrs: Dict[str, bool],
        required: Dict[str, int],
        current: Dict[str, int],
        s: int,
        min_LT_needed: int,
        LT_current: int,
    ) -> Dict[str, object]:
        """Return dict of reasons if accepting person would make finishing infeasible, else {}.
        Feasibility rule (single attr): have + delta + (s-1) >= required
        Intersection LT: LT + deltaLT + (s-1) >= min_LT_needed
        """
        breaks = {}
        # Single-attribute feasibility
        for key, needed in required.items():
            have = int(current.get(key, 0))
            delta = 1 if person_attrs.get(key, False) else 0
            if have + delta + max(0, s - 1) < int(needed):
                breaks[key] = {
                    "have": have,
                    "delta": delta,
                    "slotsAfter": max(0, s - 1),
                    "needed": int(needed),
                }
        # LT intersection feasibility
        delta_LT = 1 if (person_attrs.get("techno_lover") and person_attrs.get("berlin_local")) else 0
        if LT_current + delta_LT + max(0, s - 1) < int(min_LT_needed):
            breaks["LT"] = {
                "have": int(LT_current),
                "delta": delta_LT,
                "slotsAfter": max(0, s - 1),
                "needed": int(min_LT_needed),
            }
        return breaks

    def _get_attr_count(self, key: str) -> int:
        return int(self.attr_accept_counts.get(key, 0))

    def _get_intersection_count(self, a: str, b: str) -> int:
        if not hasattr(self, "intersection_accept_counts"):
            self.intersection_accept_counts = {}
        return int(self.intersection_accept_counts.get((a, b), 0))

    def _scenario2_pop_params(self):
        if hasattr(self, "_scenario2_cache"):
            return self._scenario2_cache

        # Pull from provided attribute statistics when available (scenario-specific)
        rel = (self.attribute_statistics or {}).get("relativeFrequencies", {})
        corr = (self.attribute_statistics or {}).get("correlations", {})

        p_T = float(rel.get("techno_lover", 0.0))
        p_W = float(rel.get("well_connected", 0.0))
        p_C = float(rel.get("creative", 0.0))
        p_L = float(rel.get("berlin_local", 0.0))

        rho_LT = float(((corr.get("techno_lover", {}) or {}).get("berlin_local", 0.0)))
        # try alternate indexing if above missing
        if rho_LT == 0.0 and "berlin_local" in corr and "techno_lover" in corr["berlin_local"]:
            rho_LT = float(corr["berlin_local"]["techno_lover"])

        rho_LC = float(((corr.get("berlin_local", {}) or {}).get("creative", 0.0)))
        if rho_LC == 0.0 and "creative" in corr and "berlin_local" in corr["creative"]:
            rho_LC = float(corr["creative"]["berlin_local"])

        var_T = p_T * (1 - p_T)
        var_W = p_W * (1 - p_W)
        var_C = p_C * (1 - p_C)
        var_L = p_L * (1 - p_L)

        p_LT = p_L * p_T + rho_LT * math.sqrt(max(0.0, var_L * var_T))
        p_LT = _clip01(p_LT)
        p_CL = p_L * p_C + rho_LC * math.sqrt(max(0.0, var_L * var_C))
        p_CL = _clip01(p_CL)

        self._scenario2_cache = {
            "p": {
                "techno_lover": p_T,
                "well_connected": p_W,
                "creative": p_C,
                "berlin_local": p_L,
            },
            "p_LT": p_LT,
            "p_CL": p_CL,
        }
        return self._scenario2_cache

    def _decide_with_selected_strategy(self, person_attrs: Dict[str, bool]) -> Tuple[bool, Dict[str, object], str]:
        if self.scenario == 2:
            accept, info = self.strategy2_should_accept(person_attrs)
            return accept, info, "strategy2"
        elif self.scenario == 3:
            accept, info = self.strategy3_should_accept(person_attrs)
            return accept, info, "strategy3"
        else:
            accept, info = self.strategy1_should_accept(person_attrs)
            return accept, info, "strategy1"

    def _all_constraints_satisfied(self) -> bool:
        for attr, required in self.required_counts.items():
            if self.attr_accept_counts.get(attr, 0) < required:
                return False
        return True

    def _total_deficit(self) -> int:
        total = 0
        for attr, required in self.required_counts.items():
            have = self.attr_accept_counts.get(attr, 0)
            if have < required:
                total += (required - have)
        return total

    def run(self):
        if not self.game_id:
            self.start_new_game()

        # Initial fetch of first person (no decision yet)
        resp = self.decide_and_fetch(person_index=0, accept=None)

        while True:
            status = resp.get("status")
            if status in ("completed", "failed"):
                self.logger.write(
                    {
                        "event": status,
                        "reason": resp.get("reason"),
                        "admittedCount": resp.get("admittedCount"),
                        "rejectedCount": resp.get("rejectedCount"),
                        "attrAcceptCounts": self.attr_accept_counts,
                    }
                )
                rejected_to_show = resp.get("rejectedCount")
                if rejected_to_show is None:
                    rejected_to_show = self.rejected_count
                print(f"Game {status}. Rejected={rejected_to_show}. Reason={resp.get('reason')}")
                return 0 if status == "completed" else 1

            # running
            self.admitted_count = resp.get("admittedCount", self.admitted_count)
            self.rejected_count = resp.get("rejectedCount", self.rejected_count)
            next_person = resp.get("nextPerson") or {}
            person_index = next_person.get("personIndex")
            attributes = next_person.get("attributes", {})

            # Choose decision based on scenario (strategy scaffold for scenario 2)
            accept, debug_info, strategy_name = self._decide_with_selected_strategy(attributes)

            # Log decision with current state BEFORE applying it
            self.logger.write(
                {
                    "event": "decision",
                    "gameId": self.game_id,
                    "scenario": self.scenario,
                    "strategy": strategy_name,
                    "personIndex": person_index,
                    "admittedCountBefore": self.admitted_count,
                    "rejectedCountBefore": self.rejected_count,
                    "attributes": attributes,
                    "accept": accept,
                    "requiredCounts": self.required_counts,
                    "attrAcceptCountsBefore": self.attr_accept_counts,
                    **debug_info,
                }
            )

            # Apply decision and fetch next
            resp = self.decide_and_fetch(person_index=person_index, accept=accept)

            # Update counts after decision based on server response
            status_after = resp.get("status")
            self.admitted_count = resp.get("admittedCount", self.admitted_count)
            self.rejected_count = resp.get("rejectedCount", self.rejected_count)

            if accept and status_after == "running":
                # Only increment attribute accept counts if decision applied and game continues
                for attr, has_attr in attributes.items():
                    if has_attr and attr in self.attr_accept_counts:
                        self.attr_accept_counts[attr] += 1
                # Intersection counter for (techno_lover, berlin_local)
                if attributes.get("techno_lover") and attributes.get("berlin_local"):
                    if not hasattr(self, "intersection_accept_counts"):
                        self.intersection_accept_counts = {}
                    key = ("techno_lover", "berlin_local")
                    self.intersection_accept_counts[key] = self.intersection_accept_counts.get(key, 0) + 1
                # Scenario 2: (creative, berlin_local)
                if attributes.get("creative") and attributes.get("berlin_local"):
                    if not hasattr(self, "intersection_accept_counts"):
                        self.intersection_accept_counts = {}
                    key_cl = ("creative", "berlin_local")
                    self.intersection_accept_counts[key_cl] = self.intersection_accept_counts.get(key_cl, 0) + 1
                # Scenario 3: (german_speaker, international) and (queer_friendly, vinyl_collector)
                if attributes.get("german_speaker") and attributes.get("international"):
                    if not hasattr(self, "intersection_accept_counts"):
                        self.intersection_accept_counts = {}
                    key_gi = ("german_speaker", "international")
                    self.intersection_accept_counts[key_gi] = self.intersection_accept_counts.get(key_gi, 0) + 1
                if attributes.get("queer_friendly") and attributes.get("vinyl_collector"):
                    if not hasattr(self, "intersection_accept_counts"):
                        self.intersection_accept_counts = {}
                    key_qv = ("queer_friendly", "vinyl_collector")
                    self.intersection_accept_counts[key_qv] = self.intersection_accept_counts.get(key_qv, 0) + 1
                # Intersection counter for (creative, berlin_local)
                if attributes.get("creative") and attributes.get("berlin_local"):
                    if not hasattr(self, "intersection_accept_counts"):
                        self.intersection_accept_counts = {}
                    key_cl = ("creative", "berlin_local")
                    self.intersection_accept_counts[key_cl] = self.intersection_accept_counts.get(key_cl, 0) + 1


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal Berghain Challenge client (rule-based)")
    parser.add_argument("--scenario", type=int, default=1, help="Scenario id: 1, 2, or 3")
    parser.add_argument(
        "--player-id",
        type=str,
        default="37806fe9-bf99-413f-96dd-168c6cee2b45",
        help="Player UUID to identify you",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://berghain.challenges.listenlabs.ai",
        help="API base URL",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="/root/listen/latest_game.txt",
        help="Path to log file for the latest game",
    )
    args = parser.parse_args(argv)

    client = BerghainClient(
        base_url=args.base_url,
        player_id=args.player_id,
        scenario=args.scenario,
        log_path=args.log_file,
    )
    try:
        return client.run()
    except Exception as ex:
        # Log failure and return non-zero
        try:
            client.logger.write(
                {
                    "event": "exception",
                    "message": str(ex),
                }
            )
        except Exception:
            pass
        print(f"Fatal error: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


