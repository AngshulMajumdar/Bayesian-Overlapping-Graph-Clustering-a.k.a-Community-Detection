# Deterministic Bayesian Overlapping Graph Clustering: Feynman–Schwinger–Dyson–Bethe Inference

Rust reproduction of Tables 4–8.

```
cargo run --release            # writes results/table4.csv ... results/table8.csv
cargo run --release -- outdir  # alternative output directory
```

No external crates.

## Layout

- `src/model.rs` – Bernoulli–Gamma hurdle model, score derivatives, generator
- `src/bethe.rs` – pairwise binary sum–product in log-odds, damping, residual R(t)
- `src/micro.rs` – Section 13.1: exact enumeration, product mean field, Bethe, Region-3 (width-two junction-tree bags)
- `src/fsdb.rs` – coordinate pair closure, Algorithm 1 (point-field roots by safeguarded Newton–bisection), product mean field, exact enumeration, inclusion-mass decision
- `src/quad.rs` – Golub–Welsch Gauss–Laguerre / Gauss–Hermite and the integrated reference of Section 13.3
- `src/tables.rs` – drivers for Tables 4–8

## Settings not fixed by the paper

- RNG: xoshiro256** seeded by splitmix64 with the paper's seeds. Synthetic data for Tables 6–8 therefore differ from the paper's instances.
- SD is the sample standard deviation (n − 1).
- Edge-pair / pair MAE uses the (1,1) pair moment.
- Table 5 iterations are synchronous damped message updates starting from zero log-odds messages.
- Table 6: FSDB starts from product-prior beliefs and zero messages; product mean field uses exact expectations under the product law, sequential damped updates.
- Algorithm 1 initial fields: π = a0/(a0+b0), λ = α0/β0 with log-scale perturbations spread over ±5×10⁻⁴, ζ = 0; roots solved in ξ = log λ and ζ; the Table 7 data generator also enforces nonempty membership.
- Table 8: π_k = 0.25 and λ_k = 2.0 for all five communities (not stated in the paper), priors as in Section 13.3, 6n candidate pairs sampled uniformly without replacement, predicted labels matched to ground truth by the best permutation before node F1/Jaccard. Each run is a separate process so peak RSS (VmHWM) is per run; runtime covers Algorithm 1 plus the decision.
