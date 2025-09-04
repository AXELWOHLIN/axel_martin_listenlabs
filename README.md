# Admission Control System

This implements an optimal admission control system that maximizes acceptance rates while meeting diversity constraints using Gaussian-copula modeling and linear programming.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the system:
```bash
python main.py --base-url YOUR_API_URL --scenario 1
```

## Usage

```bash
python main.py --base-url URL --scenario {1,2,3} [options]
```

Options:
- `--player-id`: Player ID (auto-generated if not provided)
- `--seed`: Random seed (default: 42)
- `--samples`: Monte Carlo samples (default: 200,000)

## How It Works

1. **Gaussian-Copula Model**: Builds a multivariate model of constrained attributes using marginals and pairwise correlations
2. **Linear Programming**: Solves for optimal acceptance probabilities to maximize overall acceptance rate
3. **Safety Buffers**: Uses Hoeffding bounds to ensure high-probability constraint satisfaction
4. **Adaptive Control**: Re-solves the optimization problem periodically to stay on track

## Files

- `main.py`: Entry point
- `runner.py`: Main game loop
- `client.py`: API wrapper
- `model.py`: Gaussian-copula modeling
- `planner.py`: Linear programming solver
- `policy.py`: Admission control policy
- `utils.py`: Matrix utilities
