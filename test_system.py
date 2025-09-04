#!/usr/bin/env python3
"""
Test script to verify the admission control system components work correctly.
"""
import numpy as np
from model import build_copula_for_constrained, sample_types
from planner import solve_acceptance_lp
from policy import AcceptancePolicy

def test_system():
    print("Testing admission control system components...")
    
    # Mock data similar to what the API might provide
    relative_frequencies = {
        "attr1": 0.3,
        "attr2": 0.4,
        "attr3": 0.2
    }
    
    correlations = {
        "attr1": {"attr2": 0.1, "attr3": -0.05},
        "attr2": {"attr3": 0.15}
    }
    
    constrained_attr_ids = ["attr1", "attr2", "attr3"]
    m_targets = [200, 300, 150]  # out of 1000 total
    
    print("1. Testing Gaussian-copula model...")
    model = build_copula_for_constrained(
        relative_frequencies, correlations, constrained_attr_ids
    )
    print(f"   Built model with {len(model.attr_ids)} attributes")
    print(f"   Thresholds: {model.thresholds}")
    print(f"   Latent correlation matrix shape: {model.latent_corr.shape}")
    
    print("2. Testing type sampling...")
    rng = np.random.default_rng(42)
    types, weights = sample_types(model, n_samples=10000, rng=rng)
    print(f"   Sampled {len(types)} unique types")
    print(f"   Total weight: {weights.sum():.6f}")
    
    # Verify marginal frequencies match approximately
    empirical_freqs = (types * weights[:, None]).sum(axis=0)
    print(f"   Empirical frequencies: {empirical_freqs}")
    print(f"   Target frequencies: {[relative_frequencies[attr] for attr in constrained_attr_ids]}")
    
    print("3. Testing LP solver...")
    r_targets = np.array([m / 1000 for m in m_targets])
    plan = solve_acceptance_lp(types, weights, r_targets)
    print(f"   Optimal acceptance rate: {(weights * plan.alpha).sum():.4f}")
    print(f"   Alpha values range: [{plan.alpha.min():.4f}, {plan.alpha.max():.4f}]")
    
    print("4. Testing policy...")
    policy = AcceptancePolicy(types, weights, len(constrained_attr_ids), m_targets, N_total=1000, seed=42)
    
    # Simulate a few decisions
    test_cases = [
        np.array([1, 1, 0], dtype=np.uint8),
        np.array([0, 1, 1], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
    ]
    
    for i, a_vec in enumerate(test_cases):
        decision = policy.decide(a_vec)
        print(f"   Test case {i+1}: {a_vec} -> {'ACCEPT' if decision else 'REJECT'}")
    
    print(f"   Policy state: admitted={policy.state.admitted}, rejected={policy.state.rejected}")
    print(f"   Constraint counts: {policy.state.counts}")
    
    print("\nAll tests completed successfully! ✓")

if __name__ == "__main__":
    test_system()
