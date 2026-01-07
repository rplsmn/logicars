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
| Checkerboard small | 16×[9→8→4→2]=224 | [264→256×4→128→...→8]=1272 | 1496 |

**Note**: Small model was updated 2026-01-07 from 728→1496 gates (6→9 layers) after training showed insufficient capacity.

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

## Session 2026-01-07: Small Model Capacity Fix

### Problem Diagnosed

Training showed soft loss decreasing (108→64) while hard accuracy stayed at ~50% (random).

**Root Cause:** The "small" model had insufficient capacity:
- Old: 6 update layers, 728 total gates
- Reference uses: 16 update layers, 3040 gates

The shallow network learned soft approximations (values near 0.5) that minimized loss but discretized to random.

### Fix Applied

Updated `create_small_checkerboard_update()` in `checkerboard.rs`:

| Property | Old | New | Reference |
|----------|-----|-----|-----------|
| Update layers | 6 | 9 | 16 |
| 256-unit layers | 1 | 4 | 10 |
| Total gates | 728 | 1496 | 3040 |

Architecture: `264→256×4→128→64→32→16→8` (10 layer sizes, 9 transitions)

### Tests Added

- `test_small_model_layer_depth` - verifies at least 8 update layers
- Updated `test_small_checkerboard_model` - verifies at least 1000 gates

Total: 138 tests passing

### Next Steps

Re-run training with new small model:
```bash
cargo run --bin train_checkerboard --release -- --small --epochs=100 --log-interval=10
```

Expected: Training should now show hard accuracy improving above 50%.

---

## Session 2026-01-07: Critical Gradient Scaling Bug Fix

### Problem

After increasing model capacity (728→1496 gates), training still showed:
- Soft loss: 104 → 64 (converging to expected value for all-0.5 outputs)
- Hard accuracy: ~50% (random)

Analysis revealed the network outputs were stuck at exactly 0.5 (uncertain).

### Root Cause: Gradient Scaling Bug

In `backward_through_time()`, we were scaling gradients by:
```rust
let scale = 1.0 / (num_steps * num_cells * effective_channels);
// = 1 / (20 * 256 * 1) = 1/5120 = 0.000195
```

This was WRONG. The division by `num_steps` has no basis in the math:

1. In BPTT, gradients from each step **accumulate** (same parameters used at each step)
2. Only the final step contributes to loss, but gradients flow back through all steps
3. The reference implementation does NOT average over steps

**Result**: Effective learning rate was `0.05 * 0.000195 = 0.00001` - 1000x too small!

### Fix Applied

Changed gradient scaling in `training.rs`:
```rust
// OLD (WRONG):
let scale = 1.0 / (num_steps * num_cells * effective_channels);

// NEW (CORRECT):
let scale = 1.0 / (num_cells * effective_channels);
```

Now effective LR is `0.05 * 0.00390625 = 0.000195` which is 20x larger.

### Files Changed

| File | Changes |
|------|---------|
| `src/training.rs` | Removed `num_steps` from gradient scaling divisor |

### Next Steps

Re-run training - this should now converge properly:
```bash
cargo run --bin train_checkerboard --release -- --small --epochs=100 --log-interval=10
```

---

**Last Updated**: 2026-01-07

---

## Session 2026-01-07: Zero-Padding Fix & Gradient Scale=1.0

### Problem

Training still showed identical loss numbers after previous fixes. Investigation revealed two more issues:

### Issue 1: NonPeriodic Boundary Handling Bug

**Root Cause**: `get_cell()` and `get_cell_array()` in `grid.rs` called `resolve_coords()` directly, bypassing the zero-padding check in `get()`. The `neighborhood()` function uses `get_cell()`, so boundary cells were NOT getting zero-padded neighbors.

**The bug (in `get_cell`):**
```rust
pub fn get_cell(&self, x: isize, y: isize) -> Vec<f64> {
    let (x, y) = self.resolve_coords(x, y);  // Clamps to edge, doesn't return zeros!
    // ...
}
```

**Fix Applied**: Added zero-padding check to BOTH `get_cell()` and `get_cell_array()`:
```rust
pub fn get_cell(&self, x: isize, y: isize) -> Vec<f64> {
    if self.boundary == BoundaryCondition::NonPeriodic {
        if x < 0 || x >= self.width as isize || y < 0 || y >= self.height as isize {
            return vec![0.0; self.channels];
        }
    }
    let (x, y) = self.resolve_coords(x, y);
    // ...
}
```

### Issue 2: Gradient Scaling Still Too Small

**Root Cause**: We were dividing gradients by `num_cells * effective_channels` = 256, but the reference uses RAW sum loss without any averaging.

**Reference loss function:**
```python
return jax.numpy.square(y[..., 0] - train_y[..., 0]).sum()  # No division!
```

**Fix Applied**: Changed to `scale = 1.0` (no averaging):
```rust
// OLD: let scale = 1.0 / (num_cells * effective_channels);
let scale = 1.0;
```

### Issue 3: Reference .py File Has Wrong Layer Sizes

The `difflogic_ca.py` file incorrectly shows `'layers': [513, 256, ...]` for update input. The actual notebook computes it dynamically:
```python
init = n_kernels * channels * layers[-1] + channels
# = 16 * 8 * 2 + 8 = 264
```

**Our implementation correctly uses 264**, not 513.

### Debug Tools Added

- `debug_checkerboard.rs` - saves PNG images of target/seed/output for visual inspection
- Added `image` crate dependency (v0.24 for Rust 1.85 compat)
- Boundary zero-padding test: `test_neighborhood_zero_padding_c1`

### Files Changed

| File | Changes |
|------|---------|
| `src/grid.rs` | Fixed `get_cell()` and `get_cell_array()` to zero-pad OOB |
| `src/grid.rs` | Updated tests for zero-padding behavior |
| `src/training.rs` | Changed gradient scale from `1/256` to `1.0` |
| `src/bin/debug_checkerboard.rs` | Added PNG image export |
| `Cargo.toml` | Added `image = "0.24"` |
| `agents/INDEX.md` | NEW: Code navigation index for LLMs |
| `AGENTS.md` | Updated to reference INDEX.md |

### Verified Correct

- ✅ Checkerboard target pattern (2×2 squares, channel 0)
- ✅ Zero-padding now works in neighborhoods
- ✅ Loss computed on channel 0 only
- ✅ Perception output ordering (c s k)
- ✅ Update input size = 264

### UNRESOLVED: Training Still Fails

Despite all fixes, training shows IDENTICAL loss numbers. This suggests the binary may not be rebuilding properly, OR there's a fundamental issue not yet identified.

**Possible causes to investigate:**
1. Cargo caching issue - try `cargo clean && cargo build --release`
2. Wrong binary being executed - verify path
3. Deterministic RNG producing identical sequences
4. Something in forward pass not using the updated code

### Next Session Should

1. Run `cargo clean && cargo build --release` to force full rebuild
2. Add print statements to verify code changes are active
3. Check if gradients are actually flowing (non-zero)
4. Verify the training loop is actually updating weights

---

## Key Files for Next Session

| Purpose | File | Lines |
|---------|------|-------|
| Boundary handling | `src/grid.rs` | 110-135 (get, get_cell, get_cell_array) |
| Gradient scaling | `src/training.rs` | 573-580 |
| Loss computation | `src/training.rs` | 279-294 (compute_loss_channel) |
| Training binary | `src/bin/train_checkerboard.rs` | Full file |
| Code index | `agents/INDEX.md` | Full file (use this first!) |
