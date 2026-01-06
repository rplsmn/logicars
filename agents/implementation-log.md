# Logicars Implementation Log

> Compact log for LLM agents. Full history: `agents/archive/`

## Current State (2026-01-06)

**Phase 2.1 🚧 IN PROGRESS** | Checkerboard training not converging - needs investigation

### What's Done
- Phase 0: Gate primitives, backprop verified
- Phase 1.1-1.4: NGrid, Perception, Update, Training modules
- Phase 1.5: GoL 99.41% accuracy, blinker/glider work
- Phase 2.1: Checkerboard infrastructure built, training stuck at ~50%
- Phase 3.2: Hard circuit export with serialization ✅

### Code Structure
```
src/
├── grid.rs           # NGrid (C=1-128), NNeighborhood, BoundaryCondition
├── perception.rs     # PerceptionModule, PerceptionKernel, connections
├── update.rs         # UpdateModule, DiffLogicCA, trainers
├── training.rs       # TrainingLoop, TrainingConfig, sync/async
├── checkerboard.rs   # Checkerboard patterns, models, loss functions
├── circuit.rs        # HardCircuit export, serialization
├── phase_0_*.rs      # Foundation: BinaryOp, GateLayer, Circuit
├── optimizer.rs      # AdamW (β2=0.99)
└── bin/
    ├── train_gol.rs         # GoL training (soft/hard loss, --log-interval)
    └── train_checkerboard.rs # Checkerboard training (--log-interval, --epochs=N)
```

### Key Commands
```bash
cargo test --lib                                              # 135 tests
cargo run --bin train_gol --release -- --full                 # GoL full training
cargo run --bin train_checkerboard --release -- --epochs=500  # Checkerboard (long)
cargo run --bin train_checkerboard --release -- --small       # Quick test
cargo run --bin train_checkerboard --release -- --log-interval=10  # Custom logging
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
| Checkerboard full | 16×[9→8→4→2]=224 | [264→256×10→...→8]=2816 | 3040 |
| Checkerboard small | 16×[9→8→4→2]=224 | [264→256→128→...→8]=504 | 728 |

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
5. **Perception output ordering**: Must be (c s k) not (c k s) - see Session 2026-01-06

---

## Phase 2.1: Checkerboard (C=8) - BLOCKED

**Status**: Training stuck at ~50% accuracy (random). Reference achieves 100% in 17 min on GPU.

### What's Built
- ✅ Pattern generation: `create_checkerboard()`, `create_random_seed()`
- ✅ Model: 728 gates (small) / 3040 gates (full)
- ✅ Training binary with `--log-interval`, `--epochs=N` options
- ✅ Tests: 135 passing
- ✅ Perception output ordering fixed to match reference (c s k)

### Training Results

**Full Model (3040 gates) - 2026-01-06:**
```
Epoch    0: soft_loss=885, hard_loss=1091, acc=49.22%
Epoch   50: soft_loss=130, hard_loss=1001, acc=48.44%
```
- Soft loss decreases but hard loss stays high
- Accuracy stuck at ~50% (random)
- Time: ~22s/epoch (~3 hours for 500 epochs)

**Reference (JAX/GPU) - Same experiment:**
- 100% accuracy in 17 minutes
- Key difference: GPU parallelism, batch_size=2

### Root Cause Analysis

**Fixed issues:**
1. ✅ Perception output ordering was (c k s), now fixed to (c s k)

**Remaining differences from reference:**
1. **batch_size=1 vs 2**: Reference averages gradients over 2 random seeds
2. **Sequential vs parallel**: Reference uses GPU, we use single-threaded CPU
3. **Performance**: 10-15x slower than JAX GPU (expected for CPU)

### Exit Criteria
- ⬜ Pattern emerges from seed (BLOCKED)
- ⬜ Generalizes 16×16 → 64×64

### Next Actions for Next Agent Session

1. **Verify ordering fix**: Run small training to see if accuracy improves
   ```bash
   cargo run --bin train_checkerboard --release -- --small --epochs=100 --log-interval=10
   ```

2. **If still stuck at ~50%**: Implement batch training (batch_size=2)
   - Accumulate gradients over 2 random seed inputs before applying
   - This matches reference and should stabilize training

3. **If batch doesn't help**: Compare layer-by-layer outputs with reference
   - Run same input through both implementations
   - Find where outputs diverge

---

## Session 2026-01-06: Perception Ordering Fix & Test Coverage

### Branch: `fix/perception-output-ordering`

### Critical Bug Found
Perception output ordering was wrong:
- **Reference**: `rearrange(x, 'k c s -> (c s k)')` = channels × output_bits × kernels
- **Our code**: Was (c k s) = channels × kernels × output_bits

This caused the update module to receive shuffled inputs, making learning impossible.

### Changes Made
1. **perception.rs**: Fixed `forward_soft`, `forward_hard`, gradient functions
2. **train_checkerboard.rs**: Added `--log-interval=N` option
3. **Tests**: Added 7 new tests (135 total)
   - `test_perception_output_ordering_csk`
   - `test_multi_step_rollout`
   - `test_non_periodic_boundaries`
   - `test_perception_output_size_multichannel`
   - `test_checkerboard_loss_soft_values`
   - `test_checkerboard_accuracy_different_sizes`
   - `test_full_checkerboard_model_input_output_sizes`
   - `test_hard_circuit_multichannel`

### Performance Analysis
| Implementation | Hardware | Time for 500 epochs |
|----------------|----------|---------------------|
| Python/JAX | Colab GPU | 17 min |
| Rust | Intel CPU | ~3 hours |

This is expected - JAX runs parallel GPU kernels. See `agents/plan.md` Phase 4.4 for optimization roadmap.

---

## Session 2026-01-05: Hard/Soft Loss & Circuit Export

### Changes Made
1. **train_gol.rs**: Added `--log-interval=N` flag and separate soft/hard loss display
2. **circuit.rs**: Hard circuit export with JSON serialization

### Branches Created
- `feature/hard-soft-loss-separation` - train_gol improvements
- `feature/hard-circuit-export` - circuit serialization module

---

**Last Updated**: 2026-01-06
