# Performance Phase 2: Parallelize Backward Pass

## Overview

**Goal**: Parallelize the backward pass (BPTT gradient computation) using rayon, matching the forward pass pattern.

**Estimated Duration**: 30-45 minutes

**Dependencies**: Phase 1 (Release Profile) recommended but not required

**Speedup Estimate**: 2-4x (backward pass is ~50% of epoch time)

---

## Success Criteria

1. ✅ Backward pass processes cells in parallel using rayon
2. ✅ Gradient accumulation is thread-safe (no data races)
3. ✅ All existing tests pass
4. ✅ Training produces same results as sequential version
5. ✅ Measured speedup on multi-core CPU

---

## Background

### Current State

The forward pass is already parallelized (training.rs:347):
```rust
let cell_results: Vec<_> = coords.par_iter().map(|&(x, y)| {
    // Process each cell in parallel
}).collect();
```

But the backward pass is sequential (training.rs:576):
```rust
for cell_idx in 0..num_cells {  // SEQUENTIAL!
    // Process each cell
    // Accumulate gradients into shared accumulators
}
```

### Why This Matters

For checkerboard training:
- 256 cells × 20 steps × 2 batch = 10,240 backward cell computations per epoch
- Each cell backward involves perception + update gradient computation
- Sequential processing leaves cores idle

---

## Task Breakdown

### Task 2.1: Analyze Current Backward Structure

**Description**: Understand the data dependencies in `accumulate_gradients()`.

**Analysis**:

Looking at training.rs:517-676, the backward pass has these operations per cell:
1. Get output gradient for cell (read from `grid_grads`)
2. Compute update gradients (independent per cell)
3. Compute perception gradients (independent per cell)
4. Accumulate into `perception_grad_accum` and `update_grad_accum` (WRITE - needs synchronization)
5. Compute input gradients for BPTT chain (independent per cell)
6. Distribute to `prev_grid_grads` (WRITE - needs synchronization)

**Key Insight**: Steps 1-3 and 5 are independent. Steps 4 and 6 require synchronization.

**Exit Criteria**:
- [ ] Documented which operations are independent
- [ ] Identified synchronization points

---

### Task 2.2: Create Per-Cell Gradient Computation Function

**Description**: Extract the per-cell gradient computation into a pure function that returns local gradients.

**Implementation**:

```rust
// training.rs - new struct for cell gradient results
struct CellGradients {
    /// Update gradients: [layer_idx][gate_idx][16 logits]
    update_grads: Vec<Vec<[f64; 16]>>,
    /// Perception gradients: [kernel_idx][layer_idx][gate_idx][16 logits]
    perception_grads: Vec<Vec<Vec<[f64; 16]>>>,
    /// Input gradients for 9 neighborhood positions × channels
    input_grads: Vec<f64>,
    /// Cell position for distributing input gradients
    x: usize,
    y: usize,
}

impl TrainingLoop {
    /// Compute gradients for a single cell (pure function, no mutation)
    fn compute_cell_gradients(
        &self,
        cell_idx: usize,
        input_grid: &NGrid,
        activations: &GridActivations,
        grid_grads: &NGrid,
    ) -> CellGradients {
        let channels = input_grid.channels;
        let x = cell_idx % input_grid.width;
        let y = cell_idx / input_grid.width;

        // Get output gradient for this cell
        let output_grads: Vec<f64> = (0..channels)
            .map(|c| grid_grads.get(x as isize, y as isize, c))
            .collect();

        // Backprop through update module
        let update_grads = self.model.update.compute_gradients(
            &activations.perception_outputs[cell_idx],
            &activations.update_activations[cell_idx],
            &output_grads,
        );

        // Compute gradient w.r.t. perception output
        let perception_output_grads = self.compute_perception_output_grads(
            &activations.perception_outputs[cell_idx],
            &activations.update_activations[cell_idx],
            &output_grads,
        );

        // Backprop through perception module
        let perception_grads = self.model.perception.compute_gradients(
            &activations.neighborhoods[cell_idx],
            &activations.perception_activations[cell_idx],
            &perception_output_grads,
        );

        // Compute gradient w.r.t. input grid (for BPTT)
        let input_grads = self.compute_input_grads(
            &activations.neighborhoods[cell_idx],
            &activations.perception_activations[cell_idx],
            &perception_output_grads,
        );

        CellGradients {
            update_grads,
            perception_grads,
            input_grads,
            x,
            y,
        }
    }
}
```

**Tests**:
```rust
#[test]
fn test_cell_gradients_deterministic() {
    // Create small model and verify compute_cell_gradients returns same values
    // when called multiple times with same inputs
    let model = create_small_checkerboard_model();
    let config = TrainingConfig::checkerboard_sync();
    let training_loop = TrainingLoop::new(model, config);
    
    let input = create_random_seed(4, 8, &mut SimpleRng::new(42));
    let target = create_checkerboard(4, 2, 8);
    
    // Run forward to get activations
    let (output, activations) = training_loop.forward_grid_soft(&input);
    
    // Create gradient grid
    let mut grid_grads = NGrid::new(4, 4, 8, input.boundary);
    for y in 0..4 {
        for x in 0..4 {
            grid_grads.set(x, y, 0, 0.1);
        }
    }
    
    // Compute twice, should be identical
    let grads1 = training_loop.compute_cell_gradients(0, &input, &activations, &grid_grads);
    let grads2 = training_loop.compute_cell_gradients(0, &input, &activations, &grid_grads);
    
    assert_eq!(grads1.update_grads.len(), grads2.update_grads.len());
    // Compare values...
}
```

**Exit Criteria**:
- [ ] `compute_cell_gradients` function compiles
- [ ] Returns correct gradient structure
- [ ] Is a pure function (no mutation of self or arguments)

---

### Task 2.3: Implement Parallel Cell Processing

**Description**: Use rayon to process all cells in parallel, collect results, then merge.

**Implementation**:

```rust
// training.rs - modify accumulate_gradients()

fn accumulate_gradients(
    &self,
    step_grids: &[NGrid],
    step_activations: &[GridActivations],
    target: &NGrid,
    perception_grad_accum: &mut Vec<Vec<Vec<[f64; 16]>>>,
    update_grad_accum: &mut Vec<Vec<[f64; 16]>>,
) {
    let num_steps = step_activations.len();
    let num_cells = step_grids[0].num_cells();
    let channels = step_grids[0].channels;

    // Initialize dL/dgrid at final step (unchanged)
    let final_output = &step_grids[num_steps];
    let mut grid_grads = NGrid::new(/* ... */);
    // ... initialize grid_grads as before ...

    // Backpropagate through steps in reverse order
    for step in (0..num_steps).rev() {
        let input_grid = &step_grids[step];
        let activations = &step_activations[step];

        // PARALLEL: Compute all cell gradients
        let cell_grads: Vec<CellGradients> = (0..num_cells)
            .into_par_iter()
            .map(|cell_idx| {
                self.compute_cell_gradients(cell_idx, input_grid, activations, &grid_grads)
            })
            .collect();

        // SEQUENTIAL: Merge gradients (fast, just accumulation)
        let mut prev_grid_grads = NGrid::new(
            input_grid.width,
            input_grid.height,
            input_grid.channels,
            input_grid.boundary,
        );

        for cell_grad in cell_grads {
            // Accumulate update gradients
            for (layer_idx, layer_grads) in cell_grad.update_grads.iter().enumerate() {
                for (gate_idx, gate_grad) in layer_grads.iter().enumerate() {
                    for i in 0..16 {
                        update_grad_accum[layer_idx][gate_idx][i] += gate_grad[i];
                    }
                }
            }

            // Accumulate perception gradients
            for (kernel_idx, kernel_grads) in cell_grad.perception_grads.iter().enumerate() {
                for channel_grads in kernel_grads {
                    for (layer_idx, layer_grads) in channel_grads.iter().enumerate() {
                        for (gate_idx, gate_grad) in layer_grads.iter().enumerate() {
                            for i in 0..16 {
                                perception_grad_accum[kernel_idx][layer_idx][gate_idx][i] +=
                                    gate_grad[i];
                            }
                        }
                    }
                }
            }

            // Distribute input gradients to prev_grid_grads
            self.distribute_input_gradients(
                &cell_grad,
                input_grid,
                &mut prev_grid_grads,
                channels,
            );
        }

        grid_grads = prev_grid_grads;
    }
}

/// Distribute input gradients to the 9 neighborhood positions
fn distribute_input_gradients(
    &self,
    cell_grad: &CellGradients,
    input_grid: &NGrid,
    prev_grid_grads: &mut NGrid,
    channels: usize,
) {
    let neighborhood_offsets: [(isize, isize); 9] = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ];

    for (pos_idx, &(dx, dy)) in neighborhood_offsets.iter().enumerate() {
        let nx = cell_grad.x as isize + dx;
        let ny = cell_grad.y as isize + dy;

        let (nx_wrapped, ny_wrapped) = if input_grid.boundary == BoundaryCondition::Periodic {
            (
                ((nx % input_grid.width as isize) + input_grid.width as isize) as usize % input_grid.width,
                ((ny % input_grid.height as isize) + input_grid.height as isize) as usize % input_grid.height,
            )
        } else {
            if nx < 0 || nx >= input_grid.width as isize || ny < 0 || ny >= input_grid.height as isize {
                continue;
            }
            (nx as usize, ny as usize)
        };

        for c in 0..channels {
            let grad_idx = pos_idx * channels + c;
            if grad_idx < cell_grad.input_grads.len() {
                let current = prev_grid_grads.get(nx_wrapped as isize, ny_wrapped as isize, c);
                prev_grid_grads.set(nx_wrapped, ny_wrapped, c, current + cell_grad.input_grads[grad_idx]);
            }
        }
    }
}
```

**Exit Criteria**:
- [ ] `accumulate_gradients` uses `into_par_iter()`
- [ ] Compiles without errors
- [ ] No mutable borrows across parallel boundary

---

### Task 2.4: Verify Numerical Equivalence

**Description**: Ensure parallel backward produces identical gradients to sequential.

**Implementation**:

Create a test that compares sequential vs parallel:

```rust
#[test]
fn test_parallel_backward_equivalence() {
    // Run training for a few epochs with sequential backward
    // Run training for same epochs with parallel backward
    // Compare final model weights
    
    let model1 = create_small_checkerboard_model();
    let model2 = model1.clone();  // Exact same starting weights
    
    let config = TrainingConfig::checkerboard_sync();
    let mut loop1 = TrainingLoop::new(model1, config.clone());
    let mut loop2 = TrainingLoop::new(model2, config);
    
    // Use same RNG for both
    loop1.set_seed(42);
    loop2.set_seed(42);
    
    let target = create_checkerboard(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_CHANNELS);
    
    // Train both for 5 epochs
    for _ in 0..5 {
        let input = create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut SimpleRng::new(123));
        loop1.train_step(&input, &target);
        loop2.train_step(&input, &target);
    }
    
    // Compare model weights (should be identical)
    // ... compare all gate logits in both models ...
}
```

**Note**: Since we're replacing the implementation, we need to test before/after the change. Alternatively, keep both implementations temporarily and compare.

**Exit Criteria**:
- [ ] Test shows parallel produces same gradients as sequential
- [ ] Loss curves match between implementations

---

### Task 2.5: Benchmark Parallel Backward

**Description**: Measure actual speedup from parallelization.

**Implementation**:

```bash
# Before (with current sequential backward)
time cargo run --bin train_checkerboard --release -- --small --epochs=50

# After (with parallel backward)
time cargo run --bin train_checkerboard --release -- --small --epochs=50
```

Also add optional timing output:
```rust
// In accumulate_gradients, optionally time:
#[cfg(feature = "timing")]
let start = std::time::Instant::now();

// ... parallel processing ...

#[cfg(feature = "timing")]
eprintln!("[TIMING] Backward pass: {:?}", start.elapsed());
```

**Exit Criteria**:
- [ ] Speedup measured (expected 2-4x on backward portion)
- [ ] Overall epoch time reduced

---

## Final Checklist

| Task | Status | Verified By |
|------|--------|-------------|
| 2.1 Analyze dependencies | ⬜ | Documentation |
| 2.2 Extract cell gradient function | ⬜ | Unit test |
| 2.3 Implement parallel processing | ⬜ | Compiles, no data races |
| 2.4 Verify numerical equivalence | ⬜ | Equivalence test |
| 2.5 Benchmark speedup | ⬜ | Measured improvement |

---

## Implementation Notes

### Thread Safety

The parallel pattern is:
1. **Parallel compute**: Each thread computes gradients for its cells (read-only access to model/activations)
2. **Sequential merge**: Single thread accumulates all results into shared accumulators

This avoids:
- Mutex contention (no locks needed)
- Atomic operations (simple sequential merge)
- Race conditions (clear ownership)

### Memory Overhead

Parallel version temporarily stores all cell gradients:
- `CellGradients` per cell ≈ 200KB for full model
- 256 cells × 200KB = 51MB temporary allocation per step

This is acceptable given typical system memory (8GB+).

### Alternative: Parallel Accumulation with Atomics

For even more parallelism, could use atomic f64 operations:
```rust
use std::sync::atomic::{AtomicU64, Ordering};

// Convert f64 to/from bits for atomic ops
fn atomic_add_f64(atomic: &AtomicU64, val: f64) {
    let mut current = atomic.load(Ordering::Relaxed);
    loop {
        let new = f64::from_bits(current) + val;
        match atomic.compare_exchange_weak(current, new.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(x) => current = x,
        }
    }
}
```

This adds complexity and may not be faster due to contention. Deferred to future optimization.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Non-deterministic due to float accumulation order | Medium | Low | Acceptable - float addition is associative enough for training |
| Memory pressure from parallel allocation | Low | Medium | Can limit rayon thread count if needed |
| Hidden dependencies we missed | Low | High | Extensive testing before/after |

---

## Rollback Plan

Keep the sequential implementation available (commented or behind feature flag) until parallel version is proven stable.

---

## Next Phase

After Phase 2 is complete, proceed to **Performance Phase 3: Batch Parallelization** for additional speedup when batch_size > 1.
