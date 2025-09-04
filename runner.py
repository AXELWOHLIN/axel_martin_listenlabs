import numpy as np
from client import GameClient
from model import build_copula_for_constrained, sample_types
from policy import AcceptancePolicy

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
