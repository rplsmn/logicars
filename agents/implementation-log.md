# Logicars Implementation Log

> Compact log for LLM agents. Full history: `agents/archive/`

## Current State (2026-01-05)

**Phase 2.1 🚧 IN PROGRESS** | GoL complete (99.41%), Checkerboard infra ready

### What's Done
- Phase 0: Gate primitives, backprop verified
- Phase 1.1-1.4: NGrid, Perception, Update, Training modules
- Phase 1.5: GoL 99.41% accuracy, blinker/glider work
- Phase 2.1: Checkerboard infrastructure built, needs training
- Phase 3.2: Hard circuit export with serialization ✅

### Code Structure
```
src/
├── grid.rs           # NGrid (C=1-128), NNeighborhood, BoundaryCondition
├── perception.rs     # PerceptionModule, PerceptionKernel, connections
├── update.rs         # UpdateModule, DiffLogicCA, trainers
├── training.rs       # TrainingLoop, TrainingConfig, sync/async
├── checkerboard.rs   # Checkerboard patterns, models, loss functions
├── circuit.rs        # HardCircuit export, serialization (NEW)
├── phase_0_*.rs      # Foundation: BinaryOp, GateLayer, Circuit
├── optimizer.rs      # AdamW (β2=0.99)
└── bin/
    ├── train_gol.rs         # GoL training (soft/hard loss, --log-interval)
    └── train_checkerboard.rs # Checkerboard training
```

### Key Commands
```bash
cargo test --lib                                        # 127 tests
cargo run --bin train_gol --release -- --full           # GoL full training
cargo run --bin train_gol --release -- --log-interval=50  # Custom log interval
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
- Tests: 127 passing

### Exit Criteria
- ⬜ Pattern emerges from seed
- ⬜ Generalizes 16×16 → 64×64

### Next Action
Run full model: `cargo run --bin train_checkerboard --release -- --epochs 500`

---

## Session 2026-01-05: Hard/Soft Loss & Circuit Export

### Changes Made
1. **train_gol.rs**: Added `--log-interval=N` flag and separate soft/hard loss display
2. **circuit.rs**: NEW - Hard circuit export with JSON serialization
   - `HardCircuit::from_soft(model)` - converts trained model
   - `circuit.active_gate_count()` - excludes pass-through gates (A, B)
   - `circuit.save(path)` / `HardCircuit::load(path)` - persistence

### Branches Created
- `feature/hard-soft-loss-separation` - train_gol improvements
- `feature/hard-circuit-export` - circuit serialization module

### Checkerboard Training Results (Small Model)

**Command**: `cargo run --bin train_checkerboard --release -- --small --epochs 100`

| Metric | Result |
|--------|--------|
| Model | 728 gates (small) |
| Best accuracy | 57.03% |
| Final accuracy | 50.98% |
| Time | 932s (~15 min) |

**Problem observed**: Soft loss decreases (1687→78) but hard loss oscillates wildly (100↔1900). Accuracy stuck at ~50% (random).

**Diagnosis**: Soft probabilities not converging to discrete decisions. Gates remain probabilistic mixtures.

**Potential fixes to investigate**:
1. Lower learning rate (0.01 instead of 0.05)
2. Use full model (4800 gates vs 728)
3. Temperature annealing to sharpen softmax over training
4. Add `--lr=` flag for experimentation

### Background Run
Full model training started: `cargo run --bin train_checkerboard --release -- --epochs 500`

---

**Last Updated**: 2026-01-05
