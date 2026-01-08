# Logicars Implementation Log

> Compact log for LLM agents. Full history: `agents/archive/`

## Current State (2026-01-07)

**Phase 2.1 🚧 IN PROGRESS** | Batch training implemented, matching reference

### What's Done

- Phase 0: Gate primitives, backprop verified
- Phase 1.1-1.4: NGrid, Perception, Update, Training modules
- Phase 1.5: GoL 99.41% accuracy, blinker/glider work
- Phase 2.1: Checkerboard infrastructure built, **batch training added**

### Code Structure

```
src/
├── grid.rs           # NGrid (C=1-128), NNeighborhood, BoundaryCondition
├── perception.rs     # PerceptionModule, PerceptionKernel, connections
├── update.rs         # UpdateModule, DiffLogicCA, trainers
├── training.rs       # TrainingLoop, TrainingConfig, batch training, loss_channel
├── checkerboard.rs   # Checkerboard patterns, models, loss functions
├── circuit.rs        # HardCircuit export, serialization
├── phase_0_*.rs      # Foundation: BinaryOp, GateLayer, Circuit
├── optimizer.rs      # AdamW (β2=0.99)
└── bin/
    ├── train_gol.rs         # GoL training (soft/hard loss, --log-interval)
    └── train_checkerboard.rs # Checkerboard training (batch_size=2)
```

### Key Commands

```bash
cargo test --lib                                              # 143 tests
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

Despite all fixes, training shows outputs collapsing to 0.5. This has been thoroughly diagnosed:

**Session 2026-01-07: Gate Ordering Fix & Gradient Analysis**

### What Was Done

1. **Gate ordering fixed**: The `BinaryOp::ALL` array was reordered to match reference:
   - Index 3 is now `A` (pass-through) instead of `NotA`
   - This matches `DEFAULT_PASS_VALUE = 10.0` at index 3 in reference
   - Verified: `soft(0,1)=0.0004, soft(1,0)=0.9996` (correct)

2. **HardGate::new() fixed**: Now stores index in ALL, not enum discriminant

3. **Debug output added**:
   - Output statistics (min, max, avg) at each epoch
   - Gradient statistics (avg, max)
   - Dominant operation percentage across all gates

### Key Findings

**Training Dynamics Issue:**

| Epoch | Output Range | Max Gradient | Pass-Through % |
|-------|-------------|--------------|----------------|
| 0 | 0.06-0.94 | 1.30 | 100% |
| 10 | 0.16-0.84 | 2.29 | 100% |
| 20 | 0.33-0.67 | 1.88 | 100% |
| 30 | 0.46-0.54 | 0.13 | 100% |
| 40 | 0.49-0.51 | 0.05 | 100% |
| 50 | 0.49-0.51 | 0.03 | 100% |

**Root Cause Analysis:**

1. **Output collapse**: After ~30 epochs, outputs converge to 0.5
2. **Gradient death**: When outputs = 0.5, loss gradients become tiny
3. **Gate stuck**: 100% of gates remain pass-through (A) throughout training
4. **Softmax saturation**: With logits[3]=10.0, prob[3]≈0.9999, so `dprob/dlogit ≈ 0`

The problem is NOT in our code correctness - it's a training dynamics issue where:
- Pass-through gates relay inputs unchanged
- Over 20 CA steps, values wash toward 0.5 (mean)
- Gradients can't overcome the 10-point logit gap for pass-through

**Reference uses same initialization (10.0) but works.** The difference is likely:
1. JAX's autodiff handles numerical aspects differently
2. Batch training (batch_size=2) provides more gradient variance  
3. Something in the gradient flow we haven't identified

### Files Changed

| File | Changes |
|------|---------|
| `src/phase_0_1.rs` | Fixed BinaryOp::ALL ordering to match reference |
| `src/phase_0_1.rs` | Fixed ProbabilisticGate::new() to use index 3 |
| `src/circuit.rs` | Fixed HardGate::new() to store index in ALL |
| `src/training.rs` | Added debug output for gradients and gate operations |
| `src/bin/train_checkerboard.rs` | Added gate verification at startup |

### All 139 Tests Pass

```bash
cargo test --lib  # 139 passed
```

### Next Steps to Investigate

1. **Try batch training**: Train on multiple samples per step
2. **Try curriculum learning**: Start with fewer steps, increase gradually
3. **Try different initialization**: Lower pass-through logit (e.g., 5.0)
4. **Compare with GoL training**: Does GoL (1-step) also have this issue?
5. **Inspect JAX gradient values**: Run reference and compare gradient magnitudes

---

## Key Files

| Purpose | File | Lines |
|---------|------|-------|
| Gate ordering | `src/phase_0_1.rs` | 68-90 (BinaryOp::ALL) |
| Gate initialization | `src/phase_0_1.rs` | 103-119 |
| Batch training | `src/training.rs` | 355-445 (train_step_batch) |
| Training binary | `src/bin/train_checkerboard.rs` | Full file |
| Code index | `agents/INDEX.md` | Full file (use this first!) |

---

## Session 2026-01-07: Batch Training Implementation

### What Was Done

Implemented batch training to match reference implementation's `batch_size=2` for checkerboard.

### Analysis of checkerboard_loss.png

The 200-epoch plateau in hard loss is **expected behavior**:
1. Gates start with pass-through (logit=10.0, prob≈99.99%)
2. Soft loss decreases as outputs drift to 0.5 (minimizing MSE)
3. Hard loss stays high because discretization picks pass-through
4. Around epoch 200, gate logits finally escape pass-through
5. Both losses then converge together

### Changes Made

1. **training.rs**:
   - Added `batch_size` field to `TrainingConfig`
   - Added `train_step_batch()` method that accumulates gradients across samples
   - Refactored `backward_through_time()` → `accumulate_gradients()` 
   - `train_step()` now delegates to `train_step_batch(&[input])`

2. **train_checkerboard.rs**:
   - Now creates batch of `batch_size` random seeds per epoch
   - Uses `train_step_batch()` for training

3. **Batch sizes from reference**:
   - GoL: batch_size=20
   - Checkerboard sync: batch_size=2
   - Checkerboard async: batch_size=1

### Why Batch Training Matters

- Provides more gradient variance (different random seeds each sample)
- Helps escape local minima faster
- May shorten the 200-epoch plateau

### Tests Added

- `test_batch_size_in_configs` - verifies correct batch sizes
- `test_train_step_batch` - basic batch training works
- `test_train_step_batch_loss_accumulates` - loss sums across batch
- `test_train_step_equivalence` - train_step = train_step_batch with 1 sample

Total: 143 tests passing

### Next Steps

Run training to see if batch training improves convergence:

```bash
cargo run --bin train_checkerboard --release -- --small --epochs=100 --log-interval=10
```

Expected: The 200-epoch plateau may be shortened with batch_size=2.

---

## Session 2026-01-08: Deep Analysis of Gradient Cancellation

### Key Finding: Gradient Cancellation is Fundamental

**Root Cause Identified:** With checkerboard target (50% black, 50% white) and outputs converging to 0.5, gradient contributions from cells with target=0 (positive gradient) EXACTLY cancel with cells with target=1 (negative gradient).

### Detailed Analysis

**Softmax Saturation Math:**
- logits[3] = 10.0 → prob[3] ≈ 0.9993
- Gradient for other logits: `d_prob_i / d_logit_i = prob_i * (1 - prob_i) ≈ 4.5e-5`
- This is ~22,000x smaller than with uniform logits

**Gradient Cancellation:**
- With output = 0.5 for all cells:
  - Cells with target=0 → dL/d_output = +1.0
  - Cells with target=1 → dL/d_output = -1.0
- Checkerboard has exactly 128 cells of each → cancels to near-zero

**Training Dynamics Observed:**
```
Epoch  0: logits[3]=9.9946, others_avg=0.0004, gap=9.99
Epoch 10: logits[3]=9.9375, others_avg=0.0042, gap=9.93
Epoch 20: logits[3]=9.8821, others_avg=0.0081, gap=9.87
```
- Gap closing rate: ~0.006 per epoch
- Projected epochs to escape (gap < 3): ~1200 epochs

**Why Reference Works:**
1. Reference plateau is ~200 epochs, suggesting faster gap closing
2. Weight decay provides consistent 0.5% reduction per epoch
3. Eventually, softmax becomes less saturated and gradients accelerate

**What We Tried:**
- Lower init (5.0): Worse! Outputs converge to 0.5 immediately
- Higher init (10.0): Keeps input variance but tiny gradients

### Debug Output Added

```rust
[DEBUG iter=N] logits: pass-through[3]=X.XX, others_avg=X.XX
```

This tracks the logit evolution to verify training progress.

### Recommendations

1. **Run longer training** (500+ epochs) to match reference
2. **Verify reference has same issue** by running notebook
3. **Potential fixes to try:**
   - Add small noise to break gradient symmetry
   - Use curriculum learning (fewer steps initially)
   - Try different loss functions

### Files Changed

| File | Changes |
|------|---------|
| `src/training.rs` | Added logit tracking debug output |

---

**Last Updated**: 2026-01-08
