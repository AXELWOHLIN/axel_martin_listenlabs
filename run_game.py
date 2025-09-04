#!/usr/bin/env python3
from runner import run_game

# Your specific parameters
base_url = "https://berghain.challenges.listenlabs.ai"
scenario = 3
player_id = "cf107033-1b58-4b18-9525-cf87da732920"
seed = 42
samples = 200_000

print(f"Starting game with scenario {scenario}")
print(f"Player ID: {player_id}")
print(f"Base URL: {base_url}")
print("-" * 50)

try:
    result = run_game(
        base_url=base_url,
        scenario=scenario,
        player_id=player_id,
        seed=seed,
        samples=samples
    )
    
    print("Game completed!")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'completed':
        print(f"Rejected count: {result['rejectedCount']}")
        print("SUCCESS: All constraints satisfied!")
    elif result['status'] == 'failed':
        print(f"Reason: {result['reason']}")
        if 'rejectedCount' in result:
            print(f"Rejected count: {result['rejectedCount']}")
    
except Exception as e:
    print(f"Error running game: {e}")
    import traceback
    traceback.print_exc()
