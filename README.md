# Logicars - Differentiable Logic Cellular Automata

A Rust implementation of [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/) from Google Research.

## Quick Start

```bash
# Build the project
cargo build --release

# Run all unit tests
cargo test --lib
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