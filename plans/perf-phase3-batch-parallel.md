# Performance Phase 3: Batch Parallelization

## Overview

**Goal**: Process batch samples in parallel during `train_step_batch()` for additional speedup.

**Estimated Duration**: 20-30 minutes

**Dependencies**: Phase 2 (Parallelize Backward) recommended

**Speedup Estimate**: 1.5-2x (when batch_size > 1)

---

## Success Criteria

1. ✅ Batch samples processed in parallel using rayon
2. ✅ Gradient accumulation across samples is thread-safe
3. ✅ All existing tests pass
4. ✅ Training produces same results as sequential batch processing
5. ✅ Measured speedup with batch_size=2

---

## Background

### Current State

The current `train_step_batch()` processes samples sequentially (training.rs:~420):

```rust
for (input, _) in batch.iter() {
    // Forward pass for this sample
    let (step_grids, step_activations, sample_soft_loss) = 
        self.forward_soft_rollout(input, num_steps);
    
    // Accumulate gradients for this sample
    self.accumulate_gradients(
        &step_grids,
        &step_activations,
        target,
        &mut perception_grad_accum,
        &mut update_grad_accum,
    );
    
    total_soft_loss += sample_soft_loss;
}
```

### Why This Matters

For checkerboard training:
- batch_size = 2
- Each sample involves 20 forward steps + backward pass
- Processing samples in parallel could nearly double throughput

With Phase 2 done, we already have per-cell parallelism. This adds sample-level parallelism on top.

---

## Task Breakdown

### Task 3.1: Create Per-Sample Gradient Computation Function

**Description**: Extract the per-sample processing into a pure function that returns sample gradients.

**Implementation**:

```rust
// training.rs - new struct for sample gradient results
struct SampleGradients {
    /// Perception gradients for this sample
    perception_grads: Vec<Vec<Vec<[f64; 16]>>>,
    /// Update gradients for this sample
    update_grads: Vec<Vec<[f64; 16]>>,
    /// Soft loss for this sample
    soft_loss: f64,
    /// Hard loss for this sample  
    hard_loss: f64,
}

impl TrainingLoop {
    /// Process one sample and return its gradients (no mutation of model)
    fn compute_sample_gradients(
        &self,
        input: &NGrid,
        target: &NGrid,
        num_steps: usize,
    ) -> SampleGradients {
        // Forward pass (stores all activations)
        let (step_grids, step_activations, soft_loss) = 
            self.forward_soft_rollout(input, num_steps);
        
        // Compute hard loss for monitoring
        let hard_output = self.run_steps(input, num_steps);
        let hard_loss = match self.config.loss_channel {
            Some(c) => Self::compute_loss_channel(&hard_output, target, c),
            None => Self::compute_loss(&hard_output, target),
        };
        
        // Create local gradient accumulators
        let mut perception_grads = self.create_perception_grad_accum();
        let mut update_grads = self.create_update_grad_accum();
        
        // Accumulate gradients for this sample
        self.accumulate_gradients(
            &step_grids,
            &step_activations,
            target,
            &mut perception_grads,
            &mut update_grads,
        );
        
        SampleGradients {
            perception_grads,
            update_grads,
            soft_loss,
            hard_loss,
        }
    }
}
```

**Exit Criteria**:
- [ ] `compute_sample_gradients` function compiles
- [ ] Returns correct gradient structure
- [ ] Is a pure function (no mutation of self)

---

### Task 3.2: Implement Parallel Sample Processing

**Description**: Use rayon to process batch samples in parallel.

**Implementation**:

```rust
// training.rs - modify train_step_batch()

pub fn train_step_batch(&mut self, batch: &[NGrid], target: &NGrid) -> (f64, f64) {
    let num_steps = self.config.num_steps;
    let batch_size = batch.len();
    
    // PARALLEL: Process all samples and collect their gradients
    let sample_results: Vec<SampleGradients> = batch
        .par_iter()
        .map(|input| self.compute_sample_gradients(input, target, num_steps))
        .collect();
    
    // SEQUENTIAL: Merge all sample gradients into final accumulators
    let mut perception_grad_accum = self.create_perception_grad_accum();
    let mut update_grad_accum = self.create_update_grad_accum();
    let mut total_soft_loss = 0.0;
    let mut total_hard_loss = 0.0;
    
    for sample_grad in sample_results {
        // Add sample gradients to accumulators
        Self::merge_perception_gradients(&mut perception_grad_accum, &sample_grad.perception_grads);
        Self::merge_update_gradients(&mut update_grad_accum, &sample_grad.update_grads);
        total_soft_loss += sample_grad.soft_loss;
        total_hard_loss += sample_grad.hard_loss;
    }
    
    // Apply accumulated gradients
    let scale = 1.0;
    self.apply_gradients(&perception_grad_accum, &update_grad_accum, scale);
    
    self.iteration += 1;
    (total_soft_loss, total_hard_loss)
}

/// Merge sample perception gradients into accumulator
fn merge_perception_gradients(
    accum: &mut Vec<Vec<Vec<[f64; 16]>>>,
    sample: &Vec<Vec<Vec<[f64; 16]>>>,
) {
    for (k, kernel_grads) in sample.iter().enumerate() {
        for (l, layer_grads) in kernel_grads.iter().enumerate() {
            for (g, gate_grad) in layer_grads.iter().enumerate() {
                for i in 0..16 {
                    accum[k][l][g][i] += gate_grad[i];
                }
            }
        }
    }
}

/// Merge sample update gradients into accumulator
fn merge_update_gradients(
    accum: &mut Vec<Vec<[f64; 16]>>,
    sample: &Vec<Vec<[f64; 16]>>,
) {
    for (l, layer_grads) in sample.iter().enumerate() {
        for (g, gate_grad) in layer_grads.iter().enumerate() {
            for i in 0..16 {
                accum[l][g][i] += gate_grad[i];
            }
        }
    }
}
```

**Exit Criteria**:
- [ ] `train_step_batch` uses `par_iter()` for samples
- [ ] Compiles without errors
- [ ] No mutable borrows across parallel boundary

---

### Task 3.3: Handle Thread-Local RNG

**Description**: Ensure random operations (if any) in forward pass are thread-safe.

**Analysis**:

The forward pass is deterministic given the model weights and input. The only randomness is:
1. Creating random seeds (done before `train_step_batch`)
2. Async training fire rate masking (not used in checkerboard sync)

For checkerboard sync training, no RNG is needed in the forward/backward pass, so this is not an issue.

**If async training needs parallel support later**:
```rust
// Use thread-local RNG
use rand::thread_rng;

// In parallel section:
let mut rng = thread_rng();
```

**Exit Criteria**:
- [ ] Verified no RNG usage in forward/backward path for sync training
- [ ] Documented approach for async training (future)

---

### Task 3.4: Verify Numerical Equivalence

**Description**: Ensure parallel batch produces identical results to sequential.

**Implementation**:

```rust
#[test]
fn test_parallel_batch_equivalence() {
    let model1 = create_small_checkerboard_model();
    let model2 = model1.clone();
    
    let config = TrainingConfig::checkerboard_sync();
    let mut loop1 = TrainingLoop::new(model1, config.clone());
    let mut loop2 = TrainingLoop::new(model2, config);
    
    let target = create_checkerboard(
        CHECKERBOARD_GRID_SIZE,
        CHECKERBOARD_SQUARE_SIZE,
        CHECKERBOARD_CHANNELS,
    );
    
    // Create batch with deterministic seeds
    let mut rng = SimpleRng::new(42);
    let batch: Vec<NGrid> = (0..2)
        .map(|_| create_random_seed(CHECKERBOARD_GRID_SIZE, CHECKERBOARD_CHANNELS, &mut rng))
        .collect();
    
    // Train both (loop2 will use parallel, loop1 sequential if we keep both)
    let (loss1_soft, loss1_hard) = loop1.train_step_batch(&batch, &target);
    let (loss2_soft, loss2_hard) = loop2.train_step_batch(&batch, &target);
    
    // Losses should be identical (same computation, just different order)
    assert!((loss1_soft - loss2_soft).abs() < 1e-10);
    assert!((loss1_hard - loss2_hard).abs() < 1e-10);
    
    // Compare a sample of gate logits
    let gate1 = &loop1.model.perception.kernels[0].layers[0].gates[0];
    let gate2 = &loop2.model.perception.kernels[0].layers[0].gates[0];
    for i in 0..16 {
        assert!(
            (gate1.logits[i] - gate2.logits[i]).abs() < 1e-8,
            "Logit {} differs: {} vs {}",
            i, gate1.logits[i], gate2.logits[i]
        );
    }
}
```

**Exit Criteria**:
- [ ] Test passes showing numerical equivalence
- [ ] Loss values match exactly

---

### Task 3.5: Benchmark Parallel Batch

**Description**: Measure actual speedup from batch parallelization.

**Implementation**:

```bash
# With batch_size=2 (default for checkerboard)
time cargo run --bin train_checkerboard --release -- --small --epochs=50

# Compare to Phase 2 baseline
```

For larger batch sizes (future), the speedup should scale with batch size up to the number of cores.

**Exit Criteria**:
- [ ] Speedup measured for batch_size=2
- [ ] Expected: 1.3-1.8x improvement

---

### Task 3.6: Add Configurable Parallelism Level

**Description**: Allow controlling parallelism for debugging or resource management.

**Implementation**:

```rust
// In TrainingConfig
pub struct TrainingConfig {
    // ... existing fields ...
    
    /// Parallel samples in batch (default: true)
    pub parallel_batch: bool,
}

// In train_step_batch
if self.config.parallel_batch && batch.len() > 1 {
    // Parallel path
    let sample_results: Vec<SampleGradients> = batch
        .par_iter()
        .map(|input| self.compute_sample_gradients(input, target, num_steps))
        .collect();
    // ...
} else {
    // Sequential path (for debugging or single-sample)
    for input in batch {
        // ...
    }
}
```

**Exit Criteria**:
- [ ] `parallel_batch` config option added
- [ ] Can disable parallelism for debugging

---

## Final Checklist

| Task | Status | Verified By |
|------|--------|-------------|
| 3.1 Extract sample gradient function | ⬜ | Compiles |
| 3.2 Implement parallel processing | ⬜ | Uses par_iter() |
| 3.3 Handle RNG (if needed) | ⬜ | No RNG in hot path |
| 3.4 Verify numerical equivalence | ⬜ | Equivalence test passes |
| 3.5 Benchmark speedup | ⬜ | Measured improvement |
| 3.6 Add parallelism config | ⬜ | Can enable/disable |

---

## Implementation Notes

### Nested Parallelism

With both Phase 2 (cell parallelism) and Phase 3 (sample parallelism), we have nested parallel regions:

```
Epoch
└── Batch (parallel samples)      <- Phase 3
    └── Sample
        └── Step
            └── Cells (parallel)  <- Phase 2
```

Rayon handles nested parallelism well via work-stealing, but:
- Total threads = `RAYON_NUM_THREADS` (default: num CPUs)
- Inner parallelism won't spawn more threads than available
- Work is distributed across all threads efficiently

### Memory Overhead

Each sample in parallel holds:
- Full activation history (20 steps × 256 cells × activations)
- Gradient accumulators

For batch_size=2: ~2x memory usage during training step.

For larger batches, may need to limit parallelism:
```rust
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

// Limit to N parallel samples
ThreadPoolBuilder::new()
    .num_threads(4)
    .build_global()
    .unwrap();
```

### Combining with Phase 2

If both Phase 2 and Phase 3 are implemented, the speedups should combine:
- Phase 2: 2-4x (cell parallelism)
- Phase 3: 1.5-2x (sample parallelism with batch=2)
- Combined: 3-8x theoretical maximum

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Memory pressure with large batches | Medium | Medium | Limit parallel samples to available memory |
| Thread contention between phases | Low | Low | Rayon handles work-stealing well |
| Non-deterministic float accumulation | Medium | Low | Acceptable for training |

---

## Rollback Plan

Add `parallel_batch: bool` config option. If issues arise, set to `false` to revert to sequential behavior without code changes.

---

## Future Enhancements

### Larger Batch Sizes

Reference uses different batch sizes:
- GoL: batch_size=20
- Checkerboard sync: batch_size=2
- Checkerboard async: batch_size=1

With efficient parallel batch, could experiment with larger batches for:
- Better gradient estimates
- More GPU-like throughput

### Gradient Accumulation

For very large effective batch sizes without memory pressure:
```rust
// Process in chunks
for chunk in batch.chunks(4) {
    // Parallel process chunk
    // Accumulate gradients
}
// Apply once at end
```

---

## Next Steps

After Phase 3, consider:
1. **Cache softmax probabilities** (Phase 4) - reduces redundant computation
2. **f32 precision** (Phase 5) - reduces memory bandwidth
3. **Profile to identify next bottleneck** - flamegraph analysis
