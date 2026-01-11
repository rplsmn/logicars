# Phase 2.2: Checkerboard Async Training

## Goal

Implement asynchronous training with fire rate masking to demonstrate **self-healing** capability.

## Why This Matters

Sync training proves pattern generation. Async training proves **robust recovery from damage**:
- Sync-trained models recover slowly from cell damage
- Async-trained models recover much faster
- This is a core scientific claim of the paper (see sync vs async comparison plot)

## Reference Implementation

From `reference/diffLogic_CA.ipynb`:

```python
def run_async(grid, params, wires, training, periodic, key):
  patches = get_grid_patches(...)
  x_new = v_run_circuit_patched(patches, params, wires, training)
  x_new = x_new.reshape(*grid.shape)
  # Fire rate mask: bernoulli per cell (first channel shape, broadcast)
  update_mask_f32 = (
      jax.random.uniform(key, x_new[..., :1].shape) <= FIRE_RATE
  ).astype(jax.numpy.float32)
  # Differentiable blend: old * (1-mask) + new * mask
  x = grid * (1 - update_mask_f32) + x_new * update_mask_f32
  return x
```

Key insight: The mask is per-cell (not per-channel), broadcasted to all channels.

## Implementation Tasks

### Task 1: Add fire rate masking to soft forward pass

Modify `forward_grid_soft` (or add `forward_grid_soft_async`) to:
1. Compute all new cell states (existing logic)
2. Generate fire rate mask per cell
3. Blend: `output[x,y] = old[x,y] * (1-mask) + new[x,y] * mask`

The mask is constant during backprop (no mask gradients needed).

### Task 2: Update backward pass for async

The backward pass already works on `step_activations`. For masked cells:
- If mask=0 (cell didn't fire): gradients should be zero for that cell
- If mask=1 (cell fired): gradients flow normally

This requires storing the mask and applying it during backward.

### Task 3: Add unit tests

- `test_async_forward_fires_partial_cells` - verify ~60% cells update
- `test_async_backward_propagates_only_fired` - gradients only for fired cells
- `test_async_training_converges` - loss decreases over epochs

### Task 4: Create train_checkerboard_async binary

Copy `train_checkerboard.rs` and modify for:
- Use `TrainingConfig::checkerboard_async()`
- 50 steps per iteration (reference)
- batch_size=1 (reference)
- Add damage/recovery visualization after training

### Task 5: Self-healing visualization

After training:
1. Load trained model
2. Run for N steps to establish pattern
3. Damage: zero out 10x10 region
4. Continue running with async inference
5. Measure error over time (save plot/gif)

## Exit Criteria

- [ ] Pattern emerges with async updates (test in rollout)
- [ ] Model recovers from random damage (visual + error metric)
- [ ] Self-healing demonstrated (error decreases after damage)
- [ ] All unit tests pass
- [ ] Training completes successfully (may take longer than sync)

## Files to Modify

1. `src/training.rs` - Add async forward/backward
2. `src/bin/train_checkerboard_async.rs` - New binary
3. `src/bin/visualize_healing.rs` - Damage recovery visualization

## Technical Notes

### Fire rate mask storage

Store mask alongside activations:
```rust
struct GridActivations {
    // ... existing fields
    fire_mask: Option<Vec<Vec<bool>>>,  // [y][x] = did this cell fire?
}
```

### Backward with mask

In `backward_through_time`, skip cells where `fire_mask[y][x] == false`.

### Gradient scaling

Same as sync: `scale = 1.0` (raw sum, no averaging).

## Commands

```bash
# Run unit tests
cargo test --lib -- async

# Build release
cargo build --release

# Train async (will take ~1 hour)
cargo run --bin train_checkerboard_async --release

# Visualize healing
cargo run --bin visualize_healing --release
```
