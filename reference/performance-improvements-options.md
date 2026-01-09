# Performance Improvement Options for Logicars

## Current State

- **Hardware**: Intel CPU (8 hours for 1000 epochs on checkerboard)
- **Reference**: JAX on Colab GPU (17 min for 500 epochs)
- **Current parallelization**: rayon for cell-level forward pass
- **Bottleneck observation**: ~29 seconds per epoch (8 hours / 1000 epochs)

## Analysis Summary

| Optimization | Speedup Est. | Risk | Effort | Notes |
|--------------|-------------|------|--------|-------|
| Release profile tuning | 1.2-2x | Very Low | Very Low | LTO, codegen-units, target-cpu |
| Parallelize backward pass | 2-4x | Low | Low | Same pattern as forward |
| Cache softmax probabilities | 1.3-2x | Low | Low | Compute once per step, not per gate call |
| Reduce allocations | 1.2-1.5x | Medium | Medium | Pre-allocate, reuse buffers |
| SIMD gate operations | 1.5-3x | Medium | Medium | Vectorize 16 ops at once |
| Gradient checkpointing | 1x (memory) | Low | Medium | Trade compute for memory |
| Batch parallelization | 1.5-2x | Low | Low | Process batch samples in parallel |
| f32 instead of f64 | 1.5-2x | Low | Medium | Half the memory bandwidth |
| GPU (Burn framework) | 10-50x | High | High | Major rewrite |

---

## Tier 1: Quick Wins (< 1 hour, low risk)

### 1.1 Release Profile Optimization

**Speedup: 1.2-2x | Risk: Very Low | Effort: 10 min**

Current `Cargo.toml` has no release profile settings. Add:

```toml
[profile.release]
lto = "fat"           # Link-time optimization (significant for compute)
codegen-units = 1     # Better optimization, slower compile
panic = "abort"       # Smaller binary, no unwinding overhead
opt-level = 3         # Maximum optimization

[profile.release.build-override]
opt-level = 3
```

Also compile with native CPU features:
```bash
RUSTFLAGS="-C target-cpu=native" cargo run --bin train_checkerboard --release
```

**Why it works**: LTO enables cross-crate inlining (critical for small functions like `execute_soft`). Native CPU enables AVX2/AVX-512 auto-vectorization.

### 1.2 Parallelize Backward Pass

**Speedup: 2-4x | Risk: Low | Effort: 30 min**

The backward pass (`accumulate_gradients`) processes cells sequentially:

```rust
// training.rs:576 - SEQUENTIAL
for cell_idx in 0..num_cells {
    // Process each cell...
}
```

But the forward pass is parallel:

```rust
// training.rs:347 - PARALLEL
let cell_results: Vec<_> = coords.par_iter().map(...).collect();
```

**Fix**: Make backward parallel with atomic accumulation or per-thread accumulators that merge at the end.

```rust
// Per-thread gradient accumulation
let cell_grads: Vec<_> = (0..num_cells).into_par_iter().map(|cell_idx| {
    // Compute gradients for this cell (returns local accumulator)
    compute_cell_gradients(cell_idx, ...)
}).collect();

// Merge all per-cell gradients into main accumulator
for cell_grad in cell_grads {
    merge_gradients(&mut perception_grad_accum, &mut update_grad_accum, cell_grad);
}
```

**Why it works**: Backward pass is ~50% of epoch time. Parallelizing gives 2x on that portion.

### 1.3 Batch Parallelization

**Speedup: 1.5-2x | Risk: Low | Effort: 20 min**

Currently, batch samples are processed sequentially in `train_step_batch()`:

```rust
// training.rs:440
for (input, _) in batch.iter() {
    // Sequential per-sample processing
}
```

**Fix**: Process batch samples in parallel:

```rust
let sample_results: Vec<_> = batch.par_iter().map(|(input, _)| {
    // Forward and accumulate for this sample
}).collect();
```

**Why it works**: batch_size=2, so limited gain, but trivial to implement.

---

## Tier 2: Medium Effort (1-4 hours, medium risk)

### 2.1 Cache Softmax Probabilities

**Speedup: 1.3-2x | Risk: Low | Effort: 1 hour**

`execute_soft()` recomputes `probabilities()` every call:

```rust
// phase_0_1.rs:144
pub fn execute_soft(&self, a: f64, b: f64) -> f64 {
    let probs = self.probabilities();  // Computes softmax every time!
    // ...
}
```

For a 3040-gate model, with 256 cells × 20 steps × 2 batch = 10,240 forward passes per epoch. Each pass calls `execute_soft()` on every gate multiple times through layers.

**Fix**: Cache probabilities per gate, invalidate on logit update:

```rust
pub struct ProbabilisticGate {
    logits: [f64; 16],
    cached_probs: Option<[f64; 16]>,  // Cached softmax
}

impl ProbabilisticGate {
    pub fn probabilities(&self) -> [f64; 16] {
        if let Some(probs) = self.cached_probs {
            return probs;
        }
        self.compute_and_cache_probs()
    }
    
    pub fn update_logits(&mut self, ...) {
        self.cached_probs = None;  // Invalidate cache
    }
}
```

Alternative: Compute probabilities once at start of forward pass, pass them through.

### 2.2 Reduce Allocations with Buffer Pools

**Speedup: 1.2-1.5x | Risk: Medium | Effort: 2 hours**

Hot paths allocate `Vec<f64>` constantly:

- `forward_soft()` returns `Vec<f64>` for each layer
- `compute_gradients()` allocates `Vec<[f64; 16]>` per call
- `neighborhood.center()` returns `Vec<f64>`

**Fix**: Pre-allocate and reuse buffers:

```rust
struct ForwardBuffers {
    layer_outputs: Vec<Vec<f64>>,  // Pre-sized for each layer
    perception_output: Vec<f64>,
    update_activations: Vec<Vec<f64>>,
}

impl TrainingLoop {
    fn forward_with_buffers(&self, input: &NGrid, buffers: &mut ForwardBuffers) {
        // Reuse buffers instead of allocating
    }
}
```

**Risk**: More complex code, thread-local buffers needed for parallel.

### 2.3 Use f32 Instead of f64

**Speedup: 1.5-2x | Risk: Low | Effort: 2 hours**

All computation uses `f64`, but `f32` is sufficient for neural network training:
- Half the memory bandwidth
- More values fit in cache
- SIMD processes 2x more values per instruction

**Fix**: Global find-replace `f64` → `f32`, update grid storage.

**Risk**: May need adjustments for numerical stability in softmax. Test carefully.

### 2.4 SIMD Gate Operations

**Speedup: 1.5-3x | Risk: Medium | Effort: 3 hours**

Each `execute_soft()` computes 16 operations sequentially:

```rust
for (i, &op) in BinaryOp::ALL.iter().enumerate() {
    output += probs[i] * op.execute_soft(a, b);
}
```

**Fix**: Use SIMD to compute all 16 operations at once:

```rust
use std::simd::{f64x4, f64x8};

fn execute_soft_simd(&self, a: f64, b: f64) -> f64 {
    let probs = f64x8::from_slice(&self.probabilities()[0..8]);
    let ops = compute_ops_simd(a, b);  // All 16 ops vectorized
    // Horizontal sum
}
```

Requires `#![feature(portable_simd)]` or `packed_simd2` crate.

---

## Tier 3: Major Effort (days, higher risk)

### 3.1 GPU Acceleration via Burn Framework

**Speedup: 10-50x | Risk: High | Effort: 1-2 weeks**

The wgpu attempt showed that naive GPU porting is slower due to dispatch overhead. Burn framework provides:

- Automatic kernel fusion
- Built-in autodiff
- Multiple backends (wgpu, CUDA, ROCm)

**Why deferred**: CPU training hasn't converged yet. GPU won't fix training dynamics, only speed.

**Prerequisites**:
1. CPU training shows >80% accuracy
2. Algorithm correctness verified

### 3.2 Gradient Checkpointing

**Speedup: 0x (memory optimization) | Risk: Low | Effort: 4 hours**

Currently storing all 20 steps of activations. Could store fewer and recompute:

- Current: O(steps × cells × activation_size) memory
- Checkpointed: O(sqrt(steps) × ...) memory, 2x compute

**When useful**: If running out of memory on larger grids or deeper models.

---

## Recommended Implementation Order

Based on speedup/effort ratio and risk:

1. **Release profile tuning** (10 min, 1.2-2x) ← Do first, free speed
2. **Parallelize backward pass** (30 min, 2-4x) ← Biggest single win
3. **Batch parallelization** (20 min, 1.5x) ← Easy if batch>1
4. **Cache softmax** (1 hour, 1.3-2x) ← Medium payoff
5. **f32 precision** (2 hours, 1.5-2x) ← Significant, moderate effort
6. **SIMD operations** (3 hours, 1.5-3x) ← Good if platform supports AVX

**Combined theoretical max**: 1.5 × 3 × 1.5 × 1.5 × 1.75 × 2 = **~20x speedup**

Realistic expectation with items 1-4: **4-8x speedup** (8 hours → 1-2 hours for 1000 epochs)

---

## Profiling First

Before implementing, profile to confirm bottlenecks:

```bash
# Install perf (Linux)
sudo apt install linux-tools-generic

# Profile training
perf record -g cargo run --bin train_checkerboard --release -- --small --epochs=10
perf report

# Or use flamegraph (recommended - visual output)
cargo install flamegraph
RUSTFLAGS="-C target-cpu=native" cargo flamegraph --bin train_checkerboard --release -- --small --epochs=10
```

This will confirm whether:
- Backward vs forward time split
- Softmax computation overhead
- Memory allocation overhead
- Specific hot functions

### Reading Profiling Results

#### Flamegraph (flamegraph.svg)

Open the generated `flamegraph.svg` in a browser. Key things to look for:

1. **Width = time spent**: Wider bars = more time. Focus on the widest bars first.

2. **Call stack depth**: Bars stack vertically. The bottom is `main()`, top is leaf functions.

3. **Hot functions to look for**:
   - `execute_soft` / `probabilities` - gate operations (target for SIMD/caching)
   - `softmax` / `exp` - softmax computation (target for caching)
   - `compute_gradients` - backward pass
   - `forward_grid_soft` - forward pass
   - `alloc` / `dealloc` / `clone` - memory allocation overhead
   - `par_iter` / `rayon` - parallelization overhead

4. **Interpreting results**:
   - If `probabilities()` is >10% of time → cache softmax optimization worthwhile
   - If `alloc/clone` significant → buffer pooling worthwhile
   - If `execute_soft` dominant → SIMD or f32 conversion worthwhile
   - If `exp()` dominant → f32 will help (f32 exp is faster than f64)

5. **Click to zoom**: Click on any bar to zoom into that subtree.

#### perf report

```bash
perf report --no-children --sort=dso,symbol
```

Key columns:
- `Overhead` - percentage of total time
- `Symbol` - function name

Look for functions with >5% overhead as optimization targets.

#### Example interpretation

```
40%  logicars::phase_0_1::ProbabilisticGate::execute_soft
25%  logicars::phase_0_1::ProbabilisticGate::probabilities
15%  logicars::training::TrainingLoop::accumulate_gradients
10%  alloc::vec::Vec<T>::clone
```

This would suggest:
1. Gate operations are the bottleneck (40%+25% = 65%)
2. Caching probabilities would eliminate 25% of time
3. Memory allocation (clone) is 10% - buffer pooling could help
4. f32 conversion would speed up the 65% gate operations

---

## Notes for LLM Implementation

**Low hallucination risk** (clear patterns to follow):
- Release profile tuning (just config)
- Parallelize backward (copy forward pattern)
- Batch parallelization (small change)

**Medium hallucination risk** (requires careful refactoring):
- Cache softmax (invalidation logic)
- Buffer pools (lifetime management)
- f32 conversion (many files)

**High hallucination risk** (complex, novel code):
- SIMD implementation (platform-specific)
- GPU/Burn port (major rewrite)

Start with Tier 1, measure improvements, then decide on Tier 2.
