# Performance Phase 5: Softmax Caching

## Overview

**Goal**: Cache softmax probabilities in gates to avoid redundant recomputation during forward/backward passes.

**Estimated Duration**: 1-2 hours

**Dependencies**: Phases 1-3 (Release Profile, Parallel Backward, Batch Parallel) should be done first for maximum impact

**Speedup Estimate**: 1.3-2x

---

## Why Caching Matters

The `probabilities()` method is called multiple times per gate per forward pass:

1. In `execute_soft()` - to compute weighted output
2. In `compute_gradients()` - to compute softmax derivatives
3. In `dominant_operation()` - for monitoring/debugging

Each call recomputes softmax:
```rust
// phase_0_1.rs:124 - Current implementation
pub fn probabilities(&self) -> [f64; 16] {
    let max_logit = self.logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mut exp_sum = 0.0;
    let mut probs = [0.0; 16];
    for i in 0..16 {
        let exp_val = (self.logits[i] - max_logit).exp();
        probs[i] = exp_val;
        exp_sum += exp_val;
    }
    for i in 0..16 {
        probs[i] /= exp_sum;
    }
    probs
}
```

For a 3040-gate model with 256 cells × 20 steps × 2 batch samples = 10,240 forward passes per epoch. If `probabilities()` is called 3x per gate, that's ~93 million softmax computations per epoch that could be cached.

---

## Success Criteria

1. ✅ Probabilities cached per gate, recomputed only when logits change
2. ✅ Cache automatically invalidated on logit updates
3. ✅ All existing tests pass (143+ tests)
4. ✅ Training produces identical numerical results
5. ✅ Measured speedup on training benchmark

---

## Task Breakdown

### Task 5.1: Add Cache to ProbabilisticGate

**Description**: Add a cached probabilities field that's computed lazily.

**Implementation**:

```rust
// src/phase_0_1.rs
use std::cell::Cell;

/// A probabilistic gate that maintains a distribution over all 16 binary operations
#[derive(Debug, Clone)]
pub struct ProbabilisticGate {
    /// Logits for each of the 16 operations (unnormalized log probabilities)
    pub logits: [f64; 16],
    /// Cached softmax probabilities (computed lazily)
    cached_probs: Cell<Option<[f64; 16]>>,
}
```

**Note**: Using `Cell<Option<...>>` allows interior mutability for caching while keeping `&self` methods.

**Exit Criteria**:
- [ ] `cached_probs` field added
- [ ] Field initialized to `None` in constructors
- [ ] `Clone` derives correctly (may need manual impl)

---

### Task 5.2: Implement Lazy Probability Computation

**Description**: Modify `probabilities()` to use cache.

**Implementation**:

```rust
impl ProbabilisticGate {
    /// Compute softmax probabilities from logits (cached)
    pub fn probabilities(&self) -> [f64; 16] {
        // Check cache first
        if let Some(probs) = self.cached_probs.get() {
            return probs;
        }
        
        // Compute softmax
        let probs = self.compute_probabilities();
        self.cached_probs.set(Some(probs));
        probs
    }
    
    /// Internal: compute softmax without caching
    fn compute_probabilities(&self) -> [f64; 16] {
        let max_logit = self.logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut exp_sum = 0.0;
        let mut probs = [0.0; 16];
        for i in 0..16 {
            let exp_val = (self.logits[i] - max_logit).exp();
            probs[i] = exp_val;
            exp_sum += exp_val;
        }
        for i in 0..16 {
            probs[i] /= exp_sum;
        }
        probs
    }
}
```

**Exit Criteria**:
- [ ] `probabilities()` returns cached value when available
- [ ] Cache miss triggers computation and stores result
- [ ] Method still returns correct values

---

### Task 5.3: Invalidate Cache on Logit Updates

**Description**: Clear cache whenever logits are modified.

**Key places where logits change**:

1. **Optimizer updates** - `apply_gradients()` in training.rs
2. **Direct logit modification** - if any code writes to `gate.logits[i]` directly

**Implementation approach A**: Add a setter method:

```rust
impl ProbabilisticGate {
    /// Update a single logit and invalidate cache
    pub fn set_logit(&mut self, index: usize, value: f64) {
        self.logits[index] = value;
        self.cached_probs.set(None);  // Invalidate
    }
    
    /// Update all logits and invalidate cache
    pub fn set_logits(&mut self, logits: [f64; 16]) {
        self.logits = logits;
        self.cached_probs.set(None);  // Invalidate
    }
    
    /// Invalidate the probability cache (call after direct logit modification)
    pub fn invalidate_cache(&self) {
        self.cached_probs.set(None);
    }
}
```

**Implementation approach B**: Make logits private and always use setters (more invasive).

**Recommendation**: Use approach A - add invalidation methods without breaking existing code that accesses `logits` directly. Add `invalidate_cache()` calls where needed.

**Exit Criteria**:
- [ ] Cache invalidation method added
- [ ] All logit update sites call invalidation
- [ ] Optimizer correctly invalidates after gradient application

---

### Task 5.4: Update Optimizer Integration

**Description**: Ensure optimizer invalidates cache after applying gradients.

**Current flow** (training.rs):
```rust
fn apply_gradients(&mut self, perception_grads: &..., update_grads: &..., scale: f64) {
    // For each gate:
    //   gate.logits[i] += optimizer.step(...)
    // Cache NOT invalidated!
}
```

**Fix**:
```rust
fn apply_gradients(&mut self, perception_grads: &..., update_grads: &..., scale: f64) {
    // Apply perception gradients
    for (kernel_idx, kernel) in self.model.perception.kernels.iter_mut().enumerate() {
        for (layer_idx, layer) in kernel.layers.iter_mut().enumerate() {
            for (gate_idx, gate) in layer.gates.iter_mut().enumerate() {
                for i in 0..16 {
                    let grad = perception_grads[kernel_idx][layer_idx][gate_idx][i] * scale;
                    let clipped = grad.clamp(-GRADIENT_CLIP, GRADIENT_CLIP);
                    let update = self.perception_optimizers[kernel_idx][layer_idx][gate_idx].step(i, clipped);
                    gate.logits[i] += update;
                }
                gate.invalidate_cache();  // ADD THIS
            }
        }
    }
    
    // Similar for update gradients...
}
```

**Exit Criteria**:
- [ ] `invalidate_cache()` called after each gate's logits are updated
- [ ] No stale cached values after training step

---

### Task 5.5: Handle Clone Correctly

**Description**: Ensure `Clone` works correctly with cached state.

**Issue**: Default `Clone` will copy the cached value, which is correct (same logits → same probs).

**Verification**: Test that cloned gates have valid cache state:

```rust
#[test]
fn test_gate_clone_cache() {
    let gate1 = ProbabilisticGate::new();
    let _ = gate1.probabilities();  // Populate cache
    
    let gate2 = gate1.clone();
    
    // Both should return same probabilities
    assert_eq!(gate1.probabilities(), gate2.probabilities());
    
    // Modifying gate2 shouldn't affect gate1
    gate2.logits[0] = 5.0;
    gate2.invalidate_cache();
    
    assert_ne!(gate1.probabilities(), gate2.probabilities());
}
```

**Exit Criteria**:
- [ ] Clone test passes
- [ ] Cloned gates are independent

---

### Task 5.6: Verify Numerical Equivalence

**Description**: Ensure cached version produces identical results.

**Implementation**:

```rust
#[test]
fn test_cached_probabilities_correctness() {
    let gate = ProbabilisticGate::new();
    
    // First call - computes and caches
    let probs1 = gate.probabilities();
    
    // Second call - should return cached
    let probs2 = gate.probabilities();
    
    // Should be exactly equal (same memory)
    for i in 0..16 {
        assert_eq!(probs1[i], probs2[i]);
    }
    
    // Verify correctness manually
    let expected = gate.compute_probabilities();  // Direct computation
    for i in 0..16 {
        assert!((probs1[i] - expected[i]).abs() < 1e-15);
    }
}

#[test]
fn test_cache_invalidation() {
    let mut gate = ProbabilisticGate::new();
    
    let probs_before = gate.probabilities();
    
    // Modify logits
    gate.logits[0] = 5.0;
    gate.invalidate_cache();
    
    let probs_after = gate.probabilities();
    
    // Should be different
    assert!(probs_before[0] != probs_after[0]);
}
```

**Exit Criteria**:
- [ ] Cached and computed values match exactly
- [ ] Cache correctly invalidates on logit change

---

### Task 5.7: Benchmark Performance

**Description**: Measure actual speedup from caching.

**Benchmark process**:

```bash
# Before caching (baseline)
time cargo run --bin train_checkerboard --release -- --small --epochs=50

# After caching
time cargo run --bin train_checkerboard --release -- --small --epochs=50
```

**Expected improvement**: 10-30% reduction in per-epoch time, depending on how much of the runtime was softmax computation.

**Exit Criteria**:
- [ ] Baseline time recorded
- [ ] Optimized time recorded
- [ ] Speedup calculated and documented

---

## Final Checklist

| Task | Status | Verified By |
|------|--------|-------------|
| 5.1 Add cache field | ⬜ | Compiles |
| 5.2 Lazy computation | ⬜ | Unit test |
| 5.3 Invalidation methods | ⬜ | Unit test |
| 5.4 Optimizer integration | ⬜ | Training works |
| 5.5 Clone handling | ⬜ | Clone test |
| 5.6 Numerical equivalence | ⬜ | Equivalence test |
| 5.7 Benchmark | ⬜ | Measured speedup |

---

## Implementation Notes

### Why Cell<Option<...>> Instead of RefCell

- `Cell` is simpler and has no runtime borrow checking overhead
- We only need to get/set the whole array, not mutate individual elements
- `[f64; 16]` is `Copy`, so `Cell` works

### Alternative: Compute-Once Pattern

Instead of caching in the gate, could compute probabilities once at the start of each training step and pass them through:

```rust
// At start of forward pass
let all_probs: Vec<Vec<[f64; 16]>> = model.gates().map(|g| g.probabilities()).collect();

// Pass probs to execute_soft
gate.execute_soft_with_probs(a, b, &probs[gate_idx])
```

This is more invasive but guarantees single computation. The caching approach is simpler and self-contained.

### Thread Safety

The current implementation uses `Cell`, which is NOT thread-safe. This is fine because:

1. Each gate is only accessed by one thread at a time in the parallel backward pass
2. Forward pass processes different cells in parallel, each with its own gate access
3. Gradient accumulation is sequential

If truly concurrent access is needed later, use `std::sync::atomic` or restructure to pre-compute probabilities.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Stale cache bugs | Medium | High | Comprehensive tests for invalidation |
| Memory overhead | Very Low | Very Low | Only 128 bytes per gate (16 × 8 + Option overhead) |
| Thread safety | Low | Medium | Current design doesn't share gates across threads |

---

## Rollback Plan

The caching is additive. To rollback:

1. Change `probabilities()` to always compute (remove cache check)
2. Remove `invalidate_cache()` calls
3. Keep fields for ABI compatibility or remove

---

## Next Steps

After Phase 5, consider:

1. **SIMD gate operations** - Vectorize the 16-op loop in `execute_soft`
2. **Buffer pooling** - Reduce allocations in hot paths
3. **f32 precision** - Already documented in Phase 4
