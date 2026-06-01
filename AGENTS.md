# Logicars — Agent Guide

Differentiable Logic Cellular Automata in Rust. Implements Google Research's
[Differentiable Logic CA](https://google-research.github.io/self-organising-systems/difflogic-ca/):
learnable logic-gate circuits that act as CA update rules (Game of Life, multi-channel
checkerboard) and self-heal from damage.

> **Note on history:** earlier versions of this repo kept a separate `agents/` folder
> (`INDEX.md`, `plan.md`, `implementation-log.md`, `qa-review.md`) and a `plans/` tree.
> Those were removed — the project state now lives in the code, the git history, and
> this file. Don't look for them.

## Orientation (code map)

Library crate (`src/`), ~8k lines. Re-exports are in `src/lib.rs`.

| Module | Lines | Responsibility |
|--------|------:|----------------|
| `gates.rs` | ~480 | `BinaryOp` (all 16 boolean ops) and `ProbabilisticGate` — soft (`softmax`) vs hard (`argmax`) gate decoding. |
| `optimizer.rs` | ~110 | `AdamW` optimizer. |
| `grid.rs` | ~800 | `NGrid` (1–128 channel grid), `NNeighborhood`, `BoundaryCondition` (periodic / fixed). |
| `perception.rs` | ~1275 | `PerceptionModule` / `PerceptionKernel` — parallel learned perception kernels + their trainer. |
| `update.rs` | ~990 | `UpdateModule`, `DiffLogicCA` (the full model = perception + update), and trainers. |
| `training.rs` | ~2540 | `TrainingLoop`, `TrainingConfig`, `SimpleRng`. Sync + async (fire-rate) forward/backward (BPTT), loss/accuracy, eval rollouts. Largest module — most CA logic lives here. |
| `checkerboard.rs` | ~570 | Checkerboard task: model factories, seed/target generators, loss/accuracy, size constants. |
| `circuit.rs` | ~590 | `HardCircuit` export — freeze a trained soft model to discrete gates (JSON save/load). |
| `gpu.rs` | — | GPU acceleration, behind the `gpu` feature. |

Binaries (`src/bin/`):

| Binary | Purpose |
|--------|---------|
| `train_gol` | Game of Life validation training. |
| `train_checkerboard` | Checkerboard **sync** training. |
| `train_checkerboard_async` | Checkerboard **async** (fire-rate) training + self-healing/robustness demos. |
| `test_generalization` | Run a trained checkerboard model on larger grids. |
| `analyze_checkerboard` | Inspect a trained `HardCircuit` model. |
| `visualize_checkerboard` | Render an animated GIF of a checkerboard rollout. |

Reference implementation (the source of truth for intended behaviour):
`reference/difflogic_ca.py`, `reference/diffLogic_CA.ipynb`, and the paper at
`reference/research-paper/`.

## Build & test

```bash
cargo test --lib                     # all unit tests
cargo test --lib -- --nocapture      # with stdout
cargo build --release                # release binaries
RUSTFLAGS="-C target-cpu=native" cargo build --release   # + AVX2/AVX-512
```

## Key implementation details

- **Soft decoding** `softmax(weights)` during training (differentiable);
  **hard decoding** `argmax(weights)` during inference (discrete).
- **Pass-through gate** initialised to logit `10.0` for training stability.
- **AdamW**, LR `0.05`, gradient clip `100.0`.
- **Async mode** uses fire-rate masking: only a fraction of cells update per step,
  driven by `TrainingLoop`'s RNG. Evaluation must not perturb that stream — use
  `TrainingLoop::rollout_async_hard(input, num_steps, seed)`, which runs hard async
  inference from a *local* RNG so multi-seed eval is reproducible.
- The backward pass (sync + async BPTT) is **finite-difference verified** — there are
  regression tests in `training.rs`. Don't "fix" the chain rule without re-deriving FD.

## Workflow

Generic development loop (no project-specific doc files required):

1. **Branch** off `main` — `feature/`, `fix/`, or `docs/<description>`. Never commit to `main`.
2. **TDD** — write/extend unit tests first, then implement, running `cargo test --lib`
   continuously. Compare behaviour against the `reference/` implementation and the paper.
3. **Commit often** with conventional messages (WHY over WHAT), smallest working commits.
4. **Push & open a PR** (`gh` CLI) for human review when the work is complete and green.

### Long-running runs

Full training runs (`cargo run --bin train_* --release`, no `--small`) take minutes to
hours and the heavy async demos (4000-step robustness) dominate wall time. Don't block on
them — hand the exact command to the human and continue independent work while they run.
