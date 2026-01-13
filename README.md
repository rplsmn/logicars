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

## Project Structure

- `src/` - Core library code
  - `grid.rs` - N-bit grid with 1-128 channels
  - `perception.rs` - Parallel perception kernels
  - `update.rs` - Update module and DiffLogicCA
  - `training.rs` - Training loop with sync/async modes
  - `checkerboard.rs` - Checkerboard experiment
- `src/bin/` - Training binaries
- `agents/` - Development documentation
- `reference/` - Python/JAX reference implementation

## Documentation

See `agents/` folder for detailed development docs:
- `plan.md` - Development roadmap
- `implementation-log.md` - Progress and learnings
- `qa-review.md` - Quality review notes