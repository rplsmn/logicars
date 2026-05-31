# Logicars - Differentiable Logic Cellular Automata

A Rust implementation of [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/) from Google Research.

## Quick Start

```bash
# Build the project
cargo build --release

# Run all unit tests
cargo test --lib
```

### Optimized Build

For maximum performance on your specific CPU, use native CPU features:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This enables AVX2/AVX-512 vectorization if your CPU supports it.

To make this permanent for local development, create `.cargo/config.toml`:
```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

## Training Experiments

### Game of Life (C=1)

Train a differentiable logic circuit to learn Conway's Game of Life rules:

```bash
# Small model (183 gates, ~2 minutes, achieves >95% accuracy)
cargo run --bin train_gol --release -- --small

# Full model (1647 gates, takes hours)
cargo run --bin train_gol --release
```

### Checkerboard Pattern (C=8)

Train a multi-channel CA to generate a checkerboard pattern from random seeds:

```bash
# Small model (728 gates, quick test)
cargo run --bin train_checkerboard --release -- --small --epochs 100

# Full model (~4800 gates, takes hours)
cargo run --bin train_checkerboard --release -- --epochs 500
```

Additional examples (logging and saving):

```bash
# Change how often training prints/evaluates (log-interval):
# Log every 20 epochs instead of the default
cargo run --bin train_checkerboard --release -- --epochs 200 --log-interval 20

# Write training metrics to a file (append mode). Useful for resuming or plotting later:
cargo run --bin train_checkerboard --release -- --epochs 200 --log=training.csv

# Save the trained model (HardCircuit JSON) at the end of training:
cargo run --bin train_checkerboard --release -- --epochs 500 --save=checkerboard.json

# Combined example: small model, frequent logs, write log file and save final model:
cargo run --bin train_checkerboard --release -- --small --epochs 100 --log-interval 5 --log=run.csv --save=checkerboard_small.json
```

Notes:
- --log-interval=N controls how often (in epochs) the script runs hard evaluation and prints/writes metrics (default 50, 10 for --small).
- --log=FILE appends CSV-style metrics to FILE (header is written when created).
- --save=PATH writes the final HardCircuit JSON to PATH after training completes.

### Checkerboard — Async / Self-Healing (C=8)

Train a multi-channel CA with fire-rate masking so it forms the checkerboard from a
random seed and recovers from damage. The binary also runs generalization, self-healing,
and robustness demos at the end:

```bash
# Small model, quick smoke test
cargo run --bin train_checkerboard_async --release -- --small --epochs 200

# Full model (takes a long time; the post-training demos add several minutes on top)
cargo run --bin train_checkerboard_async --release -- --epochs 1000 --log=async.csv
```

Async training matches the reference protocol: constant LR (`0.05`), AdamW weight decay
`1e-2`, batch size 1, 50 steps/epoch, 14×14 grid, ~800–1100 epochs. It evaluates by
averaging hard accuracy over 16 fixed random **input seeds** (each with its own fire
order), giving a stable, comparable convergence signal for logging and early stop.

To watch the learned dynamics, render GIFs from a saved model:

```bash
cargo run --bin train_checkerboard_async --release -- --epochs 1100 --save=async_model.json
cargo run --bin visualize_async_healing --release -- async_model.json --prefix=async
# -> async_rollout.gif (14×14), async_generalize.gif (56×56), async_heal.gif (damage + regrow)
```

#### Getting async to converge: reference alignment

The async checkerboard initially drove its **soft** loss to ~0 while its **hard** (argmax)
accuracy stayed at chance (~53%). The forward and backward passes were ruled out first
(finite-difference tests confirm the gradients, and `forward_hard` is bit-identical to
`forward_soft` at saturation), which left one accidental divergence from the reference:

- **What we got wrong — fixed wiring.** The reference randomly permutes every gate layer's
  input connections (`get_unique_connections` / `get_moore_connections` each end with a
  `jax.random.permutation`). The Rust port used fixed deterministic wiring (gate *i* reads
  inputs `(2i, 2i+1)`); a comment in `perception.rs` even read *"reference uses random
  permutation … we skip that."* That structured, local graph mixes information too slowly
  for the deep 20-layer update network across a 50-step rollout — gradient descent finds a
  soft solution whose discretized argmax never aligns. A `--wiring=` ablation pinned it
  down: permuting the **update module's** connections is both necessary and sufficient
  (permuted update → ~100% hard accuracy; deterministic update → stuck at chance, even with
  perception fully permuted).

Where we **deliberately** diverge from the reference:

- **Distinct per-kernel perception wiring.** The reference shares one permuted wiring across
  all 16 perception kernels, leaving them identical (differentiated only by the update
  module's asymmetric input wiring). We draw a fresh permutation *per kernel*, so kernels get
  distinct wiring, breaking that symmetry. The ablation shows this is **not** required
  (update-permutation alone converges with identical kernels) — but it speeds the hard-accuracy
  tip-over (~epoch 300 vs ~450).
- **Evaluation metric.** The reference only plots training loss; we report mean hard accuracy
  over a fixed set of 16 random input seeds, as a stable generalization signal for logging and
  early stopping.

Wiring permutations are seeded (`CHECKERBOARD_ASYNC_WIRING_SEED = 23`, the reference's seed)
for reproducibility and apply only to the async model — the sync and GoL paths keep their
existing deterministic wiring. A `--wiring={permuted,deterministic,update-only,perception-only}`
switch on the trainer reproduces the ablation.

## Project Structure

- `src/` — core library code
  - `gates.rs` — `BinaryOp` (16 boolean ops) and `ProbabilisticGate`
  - `optimizer.rs` — AdamW
  - `grid.rs` — N-bit grid with 1–128 channels
  - `perception.rs` — parallel perception kernels
  - `update.rs` — update module and `DiffLogicCA`
  - `training.rs` — training loop with sync/async modes (BPTT, eval rollouts)
  - `checkerboard.rs` — checkerboard task (models, seeds, loss/accuracy)
  - `circuit.rs` — `HardCircuit` export (discrete model JSON)
- `src/bin/` — training/analysis binaries
- `reference/` — Python/JAX reference implementation and the paper

## Documentation

See [`AGENTS.md`](AGENTS.md) for the code map, build/test commands, and the contributor
workflow.