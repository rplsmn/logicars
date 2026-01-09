# Performance Phase 4: f32 Precision

## Overview

**Goal**: Convert all numerical computation from f64 to f32 for improved performance.

**Estimated Duration**: 2-3 hours

**Dependencies**: Profiling results should confirm gate operations are a significant bottleneck

**Speedup Estimate**: 1.5-2x

---

## Why f32 is Faster

1. **Half the memory bandwidth**: f32 is 4 bytes vs f64's 8 bytes
   - More values fit in CPU cache
   - Less memory traffic

2. **2x SIMD width**: AVX2 processes 8 f32s vs 4 f64s per instruction
   - Auto-vectorization becomes more effective

3. **Faster transcendentals**: `exp()`, `sqrt()` are faster for f32
   - Softmax uses `exp()` heavily

4. **Good enough precision**: Neural network training typically uses f32
   - Reference JAX implementation uses f32 by default

---

## Success Criteria

1. ✅ All f64 types converted to f32 in hot paths
2. ✅ All existing tests pass (may need tolerance adjustments)
3. ✅ Training produces similar results (within expected f32 precision)
4. ✅ Measured speedup of 1.3x or better

---

## Pre-Implementation Checklist

Before starting, verify with profiling that:

- [ ] Gate operations (`execute_soft`, `probabilities`) are >30% of runtime
- [ ] OR memory operations (`clone`, `alloc`) are >10% of runtime
- [ ] No obvious other bottleneck that would be easier to fix

If profiling shows something else is dominant (e.g., a single function with a bug), fix that first.

---

## Task Breakdown

### Task 4.1: Create Type Alias for Easy Switching

**Description**: Add a type alias so we can easily switch between f32/f64.

**Implementation**:

```rust
// src/lib.rs or src/types.rs
/// Floating-point type used throughout the library.
/// Change to f64 if higher precision is needed.
pub type Float = f32;

/// Constants for Float type
pub const FLOAT_EPSILON: Float = 1e-6;
```

**Rationale**: Using a type alias makes it easy to switch back if issues arise.

**Exit Criteria**:
- [ ] Type alias defined
- [ ] Can be changed to f64 in one place

---

### Task 4.2: Update Core Data Structures

**Description**: Convert grid and gate storage to f32.

**Files to modify**:

1. **`src/grid.rs`**:
   ```rust
   // Change
   data: Vec<f64>
   // To
   data: Vec<Float>
   
   // Update all method signatures
   pub fn get(...) -> Float
   pub fn set(..., value: Float)
   ```

2. **`src/phase_0_1.rs`**:
   ```rust
   // Change
   logits: [f64; 16]
   // To
   logits: [Float; 16]
   
   // Update constants
   const DEFAULT_PASS_VALUE: Float = 10.0;
   ```

3. **`src/optimizer.rs`**:
   ```rust
   // All optimizer state
   m: [f64; 16] -> m: [Float; 16]
   v: [f64; 16] -> v: [Float; 16]
   ```

**Exit Criteria**:
- [ ] NGrid uses Float
- [ ] ProbabilisticGate uses Float
- [ ] AdamW uses Float
- [ ] Code compiles

---

### Task 4.3: Update Perception and Update Modules

**Description**: Convert perception and update modules to f32.

**Files to modify**:

1. **`src/perception.rs`**:
   - GateLayer activations
   - PerceptionKernel outputs
   - PerceptionModule gradients

2. **`src/update.rs`**:
   - UpdateModule activations
   - DiffLogicCA outputs

**Pattern**:
```rust
// Find all occurrences of:
Vec<f64>
[f64; N]
f64

// Replace with:
Vec<Float>
[Float; N]
Float

// Update literals:
0.0 -> 0.0 as Float  // or just 0.0_f32 if not using alias
1.0 -> 1.0 as Float
```

**Exit Criteria**:
- [ ] perception.rs compiles with Float
- [ ] update.rs compiles with Float
- [ ] All modules integrated

---

### Task 4.4: Update Training Loop

**Description**: Convert training infrastructure to f32.

**Files to modify**:

1. **`src/training.rs`**:
   - Loss computation
   - Gradient accumulators
   - TrainingConfig constants

**Key changes**:
```rust
// Gradient accumulators
Vec<Vec<[f64; 16]>> -> Vec<Vec<[Float; 16]>>

// Loss computation
pub fn compute_loss(...) -> Float

// Constants
pub const FIRE_RATE: Float = 0.6;
pub const GRADIENT_CLIP: Float = 100.0;
```

**Exit Criteria**:
- [ ] training.rs compiles with Float
- [ ] Loss/accuracy computation works

---

### Task 4.5: Update Tests

**Description**: Adjust test tolerances for f32 precision.

**Changes needed**:

```rust
// f64 tolerance
assert!((a - b).abs() < 1e-10);

// f32 tolerance (less precise)
assert!((a - b).abs() < 1e-5);

// Or use approx crate
use approx::assert_relative_eq;
assert_relative_eq!(a, b, epsilon = 1e-5);
```

**Common issues**:
- Gradient tests may need looser tolerances
- Loss comparison tests may drift slightly
- Numerical gradient tests especially sensitive

**Exit Criteria**:
- [ ] All tests pass
- [ ] Tolerances are reasonable (not too loose)

---

### Task 4.6: Update Binaries

**Description**: Update training binaries for f32.

**Files**:
- `src/bin/train_checkerboard.rs`
- `src/bin/train_gol.rs`
- Other binaries

**Changes**: Mostly automatic if using Float type alias.

**Exit Criteria**:
- [ ] All binaries compile
- [ ] Training runs produce reasonable output

---

### Task 4.7: Benchmark and Verify

**Description**: Measure speedup and verify training still works.

**Benchmarks**:
```bash
# Before (with f64) - record baseline
time cargo run --bin train_checkerboard --release -- --small --epochs=10

# After (with f32) - measure improvement
time cargo run --bin train_checkerboard --release -- --small --epochs=10
```

**Verification**:
1. Loss values should be similar (within 1% relative)
2. Training dynamics should be similar
3. No NaN or Inf values

**Exit Criteria**:
- [ ] Speedup measured and documented
- [ ] Training produces valid results
- [ ] No numerical instability

---

## Implementation Strategy

### Option A: Type Alias (Recommended)

Use `type Float = f32;` throughout. Advantages:
- Easy to switch back to f64 if issues
- Single point of change
- Clear intent

### Option B: Direct Replacement

Find-replace `f64` → `f32`. Advantages:
- Simpler, no indirection
- Slightly clearer in code

**Recommendation**: Use Option A for safety, can simplify later.

---

## Potential Issues

### 1. Numerical Instability in Softmax

Softmax can overflow/underflow with f32. Current implementation should be safe if using max-subtraction trick:

```rust
fn softmax(logits: &[Float]) -> Vec<Float> {
    let max = logits.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
    let exps: Vec<Float> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: Float = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
```

If not using max-subtraction, add it.

### 2. Gradient Accumulation Precision

Summing many small gradients can lose precision. Mitigations:
- Kahan summation (complex, likely not needed)
- Periodic gradient normalization (already done via clipping)

### 3. Test Tolerance Failures

Some tests may fail due to reduced precision. Solution:
- Adjust tolerance from 1e-10 to 1e-5
- Use relative tolerance instead of absolute

---

## Rollback Plan

If f32 causes issues:

1. Change `type Float = f32;` to `type Float = f64;`
2. Rebuild and retest
3. Document which specific operation failed

---

## Files to Modify (Summary)

| File | Changes |
|------|---------|
| `src/lib.rs` | Add `type Float = f32;` |
| `src/grid.rs` | Convert data storage and methods |
| `src/phase_0_1.rs` | Convert gate logits and operations |
| `src/perception.rs` | Convert activations and gradients |
| `src/update.rs` | Convert activations and gradients |
| `src/training.rs` | Convert loss, gradients, accumulators |
| `src/optimizer.rs` | Convert optimizer state |
| `src/checkerboard.rs` | Convert pattern generation |
| `src/circuit.rs` | Convert hard circuit (if used) |
| `src/bin/*.rs` | Should auto-update via Float alias |
| Tests | Adjust tolerances |

---

## Estimated Line Changes

- ~50 type signature changes
- ~20 constant literal updates
- ~30 test tolerance adjustments
- Total: ~100 lines modified across ~10 files

---

## Decision Point After Profiling

If profiling shows:

| Profile Result | Recommendation |
|----------------|----------------|
| Gate ops >40% | Proceed with f32 |
| Gate ops 20-40% | Proceed, but expect modest gains |
| Gate ops <20% | Consider other optimizations first |
| Memory/alloc >15% | Consider buffer pooling instead/also |
| Single hot function | Fix that function first |

---

## Next Steps After f32

If f32 provides good speedup and more is needed:

1. **Cache softmax probabilities** - avoid recomputation
2. **SIMD gate operations** - vectorize the 16-op loop
3. **Buffer pooling** - reduce allocations

---

## References

- Rust f32 vs f64: https://doc.rust-lang.org/std/primitive.f32.html
- Mixed precision training: https://arxiv.org/abs/1710.03740
- Softmax numerical stability: https://cs231n.github.io/linear-classify/#softmax
