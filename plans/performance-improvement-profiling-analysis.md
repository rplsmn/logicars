# Profiling Analysis - Logicars Training Performance

**Date**: 2026-01-09
**Branch**: perf/phase1-release-profile
**Profiling Command**: `perf record -g cargo run --bin train_checkerboard --release`
**Total Samples**: 1,986,386,421,443

## Executive Summary

The backward pass parallelization (commit eab9068) **IS IMPLEMENTED** and working correctly (src/training.rs:678-679 uses `.into_par_iter()`). However, the backward pass still consumes **40.5% of total runtime** because the **work inside the parallel loop is expensive**:

1. **Softmax recomputation** accounts for 27.4% of total time
2. **Vector allocations** account for 20% of total time
3. The parallel work itself involves these expensive operations repeatedly

## Critical Bottlenecks (by sample count)

### 1. Backward Pass: 40.5% (Already Parallelized!)

**Key Finding**: Backward pass uses `par_iter()` but is still expensive.

```
TrainingLoop::accumulate_gradients::{{closure}}: 804,326,262,798 samples (40.5%)
├─ PerceptionModule::compute_gradients:          467,769,196,494 samples (23.5%)
│  └─ compute_kernel_gradients:                  465,656,852,704 samples (23.4%)
└─ UpdateModule::compute_gradients:              218,115,551,995 samples (11.0%)
```

**Why it's slow despite parallelization**:
- Each cell's gradient computation calls `probabilities()` repeatedly (see point #2)
- Heavy vector allocations inside the parallel work (see point #3)
- The merge phase (lines 735-791) is sequential but necessary

**Code location**: src/training.rs:678-725 (parallel), 735-791 (sequential merge)

### 2. Softmax Recomputation: 27.4% ⚠️ TOP OPTIMIZATION TARGET

**Critical Issue**: `ProbabilisticGate::probabilities()` recomputes softmax on EVERY call.

```
std::f64::exp() calls:                           ~545,000,000,000 samples (27.4%)
ProbabilisticGate::probabilities():              ~505,000,000,000 samples (25.4%)
```

**Root cause** (src/phase_0_1.rs:185):
```rust
pub fn compute_gradients(&self, ...) -> [f64; 16] {
    let probs = self.probabilities();  // ← RECOMPUTES SOFTMAX EVERY TIME
    // ...
}
```

**Impact**: For a 3040-gate model with 256 cells × 20 steps × 2 batch × multiple gates per cell:
- Forward pass: thousands of `probabilities()` calls
- Backward pass: thousands more `probabilities()` calls
- Each call does 16× `f64::exp()` + normalization

**Recommended fix**: Cache probabilities, invalidate on logit update (see performance-improvements-options.md §2.1)

### 3. Vector Allocations: 20% ⚠️ SECOND TARGET

```
Vec::from_iter / Vec::extend_trusted:            ~394,000,000,000 samples (20%)
├─ Vec::from_iter (forward hard):                159,059,508,537 samples (8.0%)
├─ Vec::extend_trusted (forward hard):           280,410,054,470 samples (14.1%)
└─ Various allocation overhead:                  ~21,000,000,000 samples (1.1%)
```

**Hot allocation paths**:
- `GateLayer::forward_hard` returns `Vec<f64>` for each layer
- `compute_gradients` allocates `Vec<[f64; 16]>` per call
- `neighborhood.center()` returns `Vec<f64>`
- Per-cell gradient structures in parallel backward pass

**Code locations**:
- src/perception.rs: GateLayer operations
- src/training.rs:678-725: Per-cell gradient allocations in parallel loop
- src/update.rs: UpdateModule operations

**Recommended fix**: Buffer pools with pre-allocated, reusable buffers (see performance-improvements-options.md §2.2)

### 4. Forward Pass: 8% (Already Efficient)

```
execute_hard / forward_hard:                     ~157,000,000,000 samples (8%)
└─ UpdateModule::forward_hard:                   159,481,000,585 samples
```

Forward pass is relatively efficient and already parallelized (cell-level).

## Revised Optimization Priorities

### ✅ Already Implemented
1. **Release profile tuning** (commit adfcede)
2. **Backward pass parallelization** (commit eab9068) ← Confirmed working!
3. **Batch parallelization** (commit 00ee30c)

### 🎯 Next Optimizations (by impact)

#### Priority 1: Cache Softmax Probabilities (1.3-2x speedup)
- **Impact**: Eliminate 27.4% of runtime
- **Effort**: 1-2 hours (medium complexity)
- **Risk**: Low (clear invalidation logic)
- **Approach**: Add `cached_probs: Option<[f64; 16]>` to `ProbabilisticGate`
- **Reference**: performance-improvements-options.md §2.1

#### Priority 2: Buffer Pools (1.2-1.5x speedup)
- **Impact**: Reduce 20% allocation overhead
- **Effort**: 2-3 hours (moderate refactoring)
- **Risk**: Medium (lifetime management, thread-local for parallel)
- **Approach**: Pre-allocate per-thread buffers for forward/backward passes
- **Reference**: performance-improvements-options.md §2.2

#### Priority 3: f32 Instead of f64 (1.5-2x speedup)
- **Impact**:
  - Half memory bandwidth → better cache utilization
  - Faster `f32::exp()` → directly speeds up the 27.4% softmax overhead
  - SIMD processes 2x more values
- **Effort**: 2-3 hours (global type change)
- **Risk**: Low (NN training typically fine with f32)
- **Approach**: Global find-replace, test numerical stability
- **Reference**: performance-improvements-options.md §2.3

## Why Parallelization Alone Wasn't Enough

The backward pass parallelization (commit eab9068) was correctly implemented but didn't achieve the expected 2-4x speedup because:

1. **Parallel speedup is limited by work cost**: If each parallel task does expensive work (softmax, allocations), parallelization only distributes that expensive work—it doesn't make it cheaper.

2. **Amdahl's Law**: Even with perfect parallelization of the cell loop:
   - 27.4% of time is spent in `exp()` (inside the parallel work)
   - 20% of time is spent in allocations (inside the parallel work)
   - Sequential merge phase (lines 735-791) still required

3. **The real bottlenecks are INSIDE the parallel loop**:
   - Each `compute_gradients()` call recomputes softmax
   - Each cell allocates temporary vectors

## Combined Speedup Estimate

If we implement priorities 1-3:
- **Softmax caching**: 1.3-2x (eliminate 27.4% overhead)
- **Buffer pools**: 1.2-1.5x (reduce 20% overhead)
- **f32 conversion**: 1.5-2x (faster exp, better cache, SIMD)

**Combined theoretical**: 1.5 × 1.3 × 1.75 = **~3.4x speedup**
**Realistic conservative**: **2-3x speedup** (accounting for Amdahl's Law)

This would bring 8-hour training down to **2.5-4 hours** for 1000 epochs.

## Profiling Details

### Flamegraph Interpretation

Key observations from flamegraph.svg:
- **Wide bars in backward pass**: Confirms 40.5% time in `accumulate_gradients`
- **Many `std::f64::exp()` call sites**: Softmax recomputation across forward AND backward
- **Rayon overhead is minimal**: Parallelization framework itself is efficient
- **alloc/Vec functions prominent**: Vector allocation hot path clearly visible

### What Parallelization Achieved

The backward pass parallelization (commit eab9068) likely gave **1.5-2x speedup** by parallelizing the 40.5% backward pass portion. Without it, backward would be ~60-80% of runtime instead of 40.5%.

**But**: We've hit diminishing returns. Further speedup requires optimizing WHAT'S INSIDE the parallel work, not the parallelism itself.

## Next Steps

1. **Implement softmax caching** (Priority 1) - biggest single win
2. **Profile again** to confirm ~27% reduction
3. **Implement buffer pools** (Priority 2) if cache locality still poor
4. **Convert to f32** (Priority 3) for additional memory bandwidth gains
5. **Profile again** to validate combined speedup

## Code References

- Backward pass parallelization: src/training.rs:678-725
- Softmax recomputation: src/phase_0_1.rs:185, src/phase_0_1.rs:149
- Vector allocations: src/perception.rs (GateLayer), src/update.rs, src/training.rs:678-725
- Performance improvement options: reference/performance-improvements-options.md
