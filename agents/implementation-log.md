# Logicars Implementation Log

> Compact log for LLM agents. Full history: `agents/archive/`

## Current State (2026-01-03)

**Phase 2.1 🚧 IN PROGRESS** | GoL complete (99.41%), Checkerboard infra ready

### What's Done
- Phase 0: Gate primitives, backprop verified
- Phase 1.1-1.4: NGrid, Perception, Update, Training modules
- Phase 1.5: GoL 99.41% accuracy, blinker/glider work
- Phase 2.1: Checkerboard infrastructure built, needs training

### Code Structure
```
src/
├── grid.rs           # NGrid (C=1-128), NNeighborhood, BoundaryCondition
├── perception.rs     # PerceptionModule, PerceptionKernel, connections
├── update.rs         # UpdateModule, DiffLogicCA, trainers
├── training.rs       # TrainingLoop, TrainingConfig, sync/async
├── checkerboard.rs   # Checkerboard patterns, models, loss functions
├── phase_0_*.rs      # Foundation: BinaryOp, GateLayer, Circuit
├── optimizer.rs      # AdamW (β2=0.99)
└── bin/
    ├── train_gol.rs         # GoL training
    └── train_checkerboard.rs # Checkerboard training
```

### Key Commands
```bash
cargo test --lib                                        # 118 tests
cargo run --bin train_gol --release -- --full           # GoL full training
cargo run --bin train_checkerboard --release            # Checkerboard (long)
cargo run --bin train_checkerboard --release -- --small # Quick test
```

---

## Technical Reference

### Hyperparameters
| Param | Value | Notes |
|-------|-------|-------|
| LR | 0.05 | |
| AdamW β2 | 0.99 | Not 0.999 - critical for convergence |
| Gradient clip | 100.0 | |
| Fire rate | 0.6 | Async training |
| Pass-through init | logit=10.0 | Gate index 3 |

### Architectures
| Model | Perception | Update | Total Gates |
|-------|------------|--------|-------------|
| GoL full | 16×[9→8→4→2→1]=240 | [17→128×10→...→1]=1407 | 1647 |
| Checkerboard | 16×[9→8→4→2]=224 | [264→256×10→...→8]=4600 | ~4800 |

### Connection Types
- `first_kernel`: center vs 8 neighbors (perception layer 1)
- `unique`: unique pair connections (all other layers)

---

## Workflow

1. Read `agents/plan.md` for requirements
2. Write tests first → implement → `cargo test --lib`
3. Verify exit criteria → update log → commit → PR

---

## Key Learnings

1. **β2=0.99** escapes local minima faster than 0.999
2. **Architecture separation**: Perception extracts features, Update decides
3. **Center cell**: Must concat, not mix into perception
4. **Per-example training slow**: Batching would help significantly

---

## Phase 2.1: Checkerboard (C=8) - IN PROGRESS

**Status**: Infrastructure complete, needs long training run

### What's Built
- Pattern generation: `create_checkerboard()`, `create_random_seed()`
- Model: 728 gates (small) / ~4800 gates (full)
- Training binary: 16×16 grid, 20-step rollout, non-periodic
- Tests: 118 passing

### Exit Criteria
- ⬜ Pattern emerges from seed
- ⬜ Generalizes 16×16 → 64×64

### Next Action
Run: `cargo run --bin train_checkerboard --release -- --epochs 500`

---

**Last Updated**: 2026-01-03
