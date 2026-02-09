# Manhattan Realistic POC

Proof-of-concept experiment for reviewer concerns about the Manhattan simulation.

## Reviewer concerns addressed

1. **Variable edge traversal times** — In the main paper we use 1 timestep per edge. Here each edge has a travel time of 1–4 steps (from OSM `length` or lane count), so longer links take longer to traverse.

2. **Queue on the edge** — Agents can be in transit on a link: at most `lanes × travel_time` agents per edge (queue capacity). Movement is still discrete-step.

3. **Safe headway / vehicle length** — At most one agent can *enter* each edge per timestep (minimum 1-step gap between entries on the same link).

## How to run

From the repo root:

```bash
python experiments/manhattan_realistic_poc.py sample_od.parquet --agents 100
```

Optional: `--max-steps 50000`, `--cache-dir .manhattan_cache`, `--seed 42`.

## What it does

- Loads the same Manhattan graph and OD pairs as the main experiment.
- **Baseline**: runs IDEAL, SP, SP-PIBT with the current model (1 step/edge, node+edge capacity only).
- **Realistic**: runs IDEAL (weighted shortest path by travel time), SP, and SP-PIBT with variable edge times, edge queue, and headway.
- Reports total delay and completion rate for both; compares overhead vs IDEAL.

## Result (conclusion)

In both settings, SP-PIBT is at least as good as SP and both stay close to IDEAL. Adding the three refinements does **not** change the relative conclusion: our method remains competitive.
