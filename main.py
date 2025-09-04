#!/usr/bin/env python3
"""
Main entry point for the admission control system.
"""
import argparse
import uuid
from runner_1 import run_game_scenario_1
from runner_2 import run_game_scenario_2
from runner_3 import run_game_scenario_3

def main():
    parser = argparse.ArgumentParser(description="Run the admission control game")
    parser.add_argument("--base-url", required=True, help="Base URL of the game API")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], default=1, help="Game scenario (1, 2, or 3)")
    parser.add_argument("--player-id", help="Player ID (if not provided, generates a new UUID)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--samples", type=int, default=200_000, help="Number of Monte Carlo samples")
    
    args = parser.parse_args()
    
    # Generate player ID if not provided
    
    print(f"Starting game with scenario {args.scenario}")
    print(f"Player ID: {args.player_id}")
    print(f"Base URL: {args.base_url}")
    print(f"Random seed: {args.seed}")
    print(f"Monte Carlo samples: {args.samples}")
    print("-" * 50)
    
    try:
        # Choose the appropriate runner based on scenario
        if args.scenario == 1:
            result = run_game_scenario_1(
                base_url=args.base_url,
                scenario=args.scenario,
                player_id=args.player_id,
                seed=args.seed,
                samples=args.samples
            )
        elif args.scenario == 2:
            result = run_game_scenario_2(
                base_url=args.base_url,
                scenario=args.scenario,
                player_id=args.player_id,
                seed=args.seed,
                samples=args.samples
            )
        elif args.scenario == 3:
            result = run_game_scenario_3(
                base_url=args.base_url,
                scenario=args.scenario,
                player_id=args.player_id,
                seed=args.seed,
                samples=args.samples
            )
        else:
            raise ValueError(f"Unknown scenario: {args.scenario}")
        
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
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
