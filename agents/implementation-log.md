# Logicars Implementation Log

> Compact log for LLM agents. Full history: `agents/archive/`

## Current State (2026-01-06)

**Phase 2.1 🚧 IN PROGRESS** | Critical bug fixed - loss now computed on channel 0 only

### What's Done

- Phase 0: Gate primitives, backprop verified
- Phase 1.1-1.4: NGrid, Perception, Update, Training modules
- Phase 1.5: GoL 99.41% accuracy, blinker/glider work
- Phase 2.1: Checkerboard infrastructure built, **critical bug fixed**

### Code Structure

```
src/
├── grid.rs           # NGrid (C=1-128), NNeighborhood, BoundaryCondition
├── perception.rs     # PerceptionModule, PerceptionKernel, connections
├── update.rs         # UpdateModule, DiffLogicCA, trainers
├── training.rs       # TrainingLoop, TrainingConfig, sync/async, loss_channel
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
cargo test --lib                                              # 137 tests
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

## Phase 2.1: Checkerboard (C=8) - BUG FIX APPLIED

**Status**: Critical bug fixed - loss was computed on ALL channels, should be channel 0 only.

### What's Built

- ✅ Pattern generation: `create_checkerboard()`, `create_random_seed()`
- ✅ Model: 728 gates (small) / 3040 gates (full)
- ✅ Training binary with `--log-interval`, `--epochs=N` options
- ✅ Tests: 137 passing
- ✅ Perception output ordering fixed to match reference (c s k)
- ✅ **Loss now computed on channel 0 only** (matching reference)

### Previous Training Results (BEFORE FIX)

**Full Model (3040 gates) - 2026-01-06:**

```
Epoch    0: soft_loss=885.2149, hard_loss=1091.0000, acc=49.22%
Epoch  200: soft_loss=64.1110, hard_loss=298.0000, acc=44.92%
Epoch  400: soft_loss=64.0139, hard_loss=238.0000, acc=50.39%
```

- Soft loss decreased but **accuracy stuck at ~50%** (random)
- This was due to loss being computed on all 8 channels instead of just channel 0

### Root Cause (FIXED 2026-01-06)

**The bug:** Our implementation computed loss on ALL 8 channels, but the target only has the pattern in channel 0. Channels 1-7 are "working memory" and should NOT be penalized.

**Reference code (difflogic_ca.py line 429):**
```python
return jax.numpy.square(y[..., 0] - train_y[..., 0]).sum()  # Channel 0 only!
```

**The fix:**
1. Added `loss_channel: Option<usize>` to `TrainingConfig`
2. Added `compute_loss_channel()` function
3. Modified `backward_through_time()` to only compute gradients for loss channel
4. Updated gradient scaling to account for single-channel loss
5. Set `loss_channel: Some(0)` for checkerboard configs

### Exit Criteria

- ⬜ Pattern emerges from seed (NEEDS RETEST)
- ⬜ Generalizes 16×16 → 64×64

### Next Steps

1. **Re-run training with fix:**
   ```bash
   cargo run --bin train_checkerboard --release -- --small --epochs=100 --log-interval=10
   ```

2. If accuracy improves above ~70%, run full training:
   ```bash
   cargo run --bin train_checkerboard --release -- --epochs=500 --log-interval=10
   ```

---

## Session 2026-01-06: Channel 0 Loss Fix

### Critical Bug Found and Fixed

**Root Cause:** Training loss was computed on all 8 channels, but only channel 0 contains the target pattern. Channels 1-7 are "working memory" - the model should be free to use them however it wants.

**Impact:** The model was being penalized for using channels 1-7 as working memory, which:
1. Made it impossible to learn the checkerboard pattern
2. Explained why soft loss decreased but hard accuracy stayed at ~50%

### Changes Made

1. **training.rs**: 
   - Added `loss_channel: Option<usize>` to `TrainingConfig`
   - Added `compute_loss_channel()` for channel-specific loss
   - Modified `train_step()` to use channel-specific loss
   - Modified `backward_through_time()` to only compute gradients for loss channel
   - Fixed gradient scaling from `num_steps * num_cells * channels` to `num_steps * num_cells * effective_channels`

2. **Tests**:
   - Added `test_compute_loss_channel` (verifies channel-specific loss)
   - Added `test_training_multichannel_channel0_loss` (verifies training works)
   - Total: 137 tests passing

### Files Changed

| File | Changes |
|------|---------|
| `src/training.rs` | Added loss_channel support, 2 new tests |
| `agents/implementation-log.md` | Updated with fix |

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

**Last Updated**: 2026-01-06
