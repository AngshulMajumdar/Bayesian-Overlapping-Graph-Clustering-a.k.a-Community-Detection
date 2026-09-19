# Deterministic Bayesian Overlapping Graph Clustering: Feynman–Schwinger–Dyson–Bethe Inference

A sample reproducibility result.

## Run

    cargo run --release

## Table 4: Exact-enumeration closure microbenchmark

| Graph  | Closure            | Singleton MAE ↓ | Edge-pair MAE ↓ |
|--------|--------------------|-----------------|-----------------|
| Path   | Product mean field | 7.622 × 10⁻²    | 3.262 × 10⁻²    |
| Path   | Bethe pair closure | < 10⁻¹²         | < 10⁻¹²         |
| Path   | Region-3           | < 10⁻¹²         | < 10⁻¹²         |
| 2-tree | Product mean field | 5.416 × 10⁻²    | 1.755 × 10⁻²    |
| 2-tree | Bethe pair closure | 1.631 × 10⁻²    | 1.435 × 10⁻³    |
| 2-tree | Region-3           | < 10⁻¹²         | < 10⁻¹²         |

## Table 5: Message residual versus exact singleton error

| Graph  | Iteration | R(t)         | Singleton MAE ↓ |
|--------|-----------|--------------|-----------------|
| Path   | 10        | 1.821 × 10⁻² | 2.112 × 10⁻³    |
| Path   | 30        | 1.105 × 10⁻⁴ | 4.594 × 10⁻⁶    |
| Path   | converged | < 10⁻¹²      | < 10⁻¹²         |
| 2-tree | 10        | 1.124 × 10⁻² | 1.685 × 10⁻²    |
| 2-tree | 27        | 9.719 × 10⁻⁵ | 1.631 × 10⁻²    |
| 2-tree | converged | < 10⁻¹²      | 1.631 × 10⁻²    |
