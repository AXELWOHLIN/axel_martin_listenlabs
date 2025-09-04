import numpy as np
from client import GameClient
from model import build_copula_for_constrained, sample_types
from policy import AcceptancePolicy

def run_game_scenario_2(base_url: str, scenario: int, player_id: str, seed: int = 42, samples: int = 200_000):
    """
    Runner specifically for Scenario 2.
    Scenario 2 format: constraints as flat dict {attribute_name: minCount}
    """
    print("Creating game client...")
    gc = GameClient(base_url)
    print("Calling new_game API...")
    meta = gc.new_game(scenario, player_id)
    print("Got game metadata for Scenario 2")

    game_id = meta["gameId"]
    constraints = meta["constraints"]  # Could be dict or array format
    stats = meta["attributeStatistics"]

    print(f"Raw constraints data: {constraints}")
    print(f"Constraints type: {type(constraints)}")

    # Handle both dict and array formats for constraints
    if isinstance(constraints, dict):
        # Dict format: {"techno_lover": 650, "well_connected": 450, ...}
        attr_ids = list(constraints.keys())
        m_targets = list(constraints.values())
    elif isinstance(constraints, list):
        # Array format: [{"attribute": "techno_lover", "minCount": 650}, ...]
        attr_ids = [c["attribute"] for c in constraints]
        m_targets = [c["minCount"] for c in constraints]
    else:
        raise ValueError(f"Unknown constraints format: {type(constraints)}")

    print(f"Constrained attributes: {attr_ids}")
    print(f"Target counts: {m_targets}")

    # Build Gaussian-copula model for those attributes only
    print("Building copula model...")
    model = build_copula_for_constrained(
        relative_frequencies=stats["relativeFrequencies"],
        correlations=stats["correlations"],
        constrained_attr_ids=attr_ids
    )

    # Monte Carlo estimate of type distribution for constrained attrs
    print(f"Sampling {samples} Monte Carlo samples...")
    rng = np.random.default_rng(seed)
    types, weights = sample_types(model, n_samples=samples, rng=rng)
    print(f"Generated {len(types)} unique attribute combinations")

    # Index map to retrieve order
    attr_index = {a: i for i, a in enumerate(model.attr_ids)}

    # Start streaming people
    print("Initializing policy...")
    policy = AcceptancePolicy(types, weights, len(attr_ids), m_targets, N_total=1000, seed=seed, block=50)

    # first call (personIndex=0) to get the first person
    print("Starting admission process...")
    resp = gc.decide_and_next(game_id, person_index=0, accept=None)

    decision_count = 0
    while resp["status"] == "running":
        person = resp["nextPerson"]
        a_vec = np.zeros(len(attr_ids), dtype=np.uint8)
        
        # Extract this person's constrained attrs in our order
        for a_id, val in person["attributes"].items():
            if a_id in attr_index:
                a_vec[attr_index[a_id]] = 1 if val else 0

        decision = policy.decide(a_vec)
        decision_count += 1
        
        if decision_count % 100 == 0:
            print(f"Processed {decision_count} people, admitted: {policy.state.admitted}, rejected: {policy.state.rejected}")
        
        resp = gc.decide_and_next(
            game_id=game_id,
            person_index=person["personIndex"],
            accept=bool(decision)
        )

    print(f"Final stats: {decision_count} total decisions")
    return resp
