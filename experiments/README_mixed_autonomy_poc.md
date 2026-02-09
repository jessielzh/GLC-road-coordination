# Manhattan Mixed Autonomy POC

Proof-of-concept: **partial compliance / mixed autonomy** — a fraction **p** of agents are "human" (non-compliant with PIBT), the rest are AVs (SP-PIBT).

## Human behavior (two models)

1. **Random yielding (`--human-model yield`):**  
   With probability `yield_prob` (default 0.15) a human waits even when they could advance on their shortest path. They do not participate in priority inheritance (AVs plan first, then humans use remaining capacity).

2. **Probabilistic gap acceptance (`--human-model gap`):**  
   When considering moving to the next node, the human moves with probability `max(0, 1 - beta * congestion_ratio)` at the target node/edge. So they are less likely to move when the slot is congested.

In both cases, if a human **attempts** to move into a node/edge that is already at capacity, we **enforce** the move as a wait and count it as a **constraint violation attempt** (proxy for near-miss or non-compliance).

## Metrics

- **Completion rate:** fraction of agents that reach their goal.
- **Total delay:** sum of arrival timesteps (travel time).
- **Violation attempts:** number of times a human tried to move into a full node/edge (capacity enforced; count is the proxy).

## How to run

From the repo root:

```bash
# Default: 100 agents, human model = random yield, sweep p = 0, 0.2, ..., 1.0
python experiments/manhattan_mixed_autonomy_poc.py sample_od.parquet --agents 100

# Custom yield probability and gap-acceptance model
python experiments/manhattan_mixed_autonomy_poc.py sample_od.parquet --agents 100 --yield-prob 0.2
python experiments/manhattan_mixed_autonomy_poc.py sample_od.parquet --agents 100 --human-model gap --gap-beta 0.5
```

Options: `--agents`, `--max-steps`, `--cache-dir`, `--seed`, `--yield-prob`, `--human-model`, `--gap-beta`.

## Output

Table with columns: **p (human fraction)**, **Total delay**, **Completion %**, **Violations**, **n_AV / n_H**.  
Shows how completion, travel time, and violation attempts change as the share of human drivers increases.
