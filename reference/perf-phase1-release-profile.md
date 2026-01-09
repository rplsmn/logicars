# Performance Phase 1: Release Profile Optimization

## Overview

**Goal**: Configure Cargo release profile for maximum CPU performance with zero code changes.

**Estimated Duration**: 10-15 minutes

**Dependencies**: None (foundational)

**Speedup Estimate**: 1.2-2x

---

## Success Criteria

1. ✅ Release profile settings added to Cargo.toml
2. ✅ LTO (Link-Time Optimization) enabled
3. ✅ Native CPU features enabled via RUSTFLAGS
4. ✅ All existing tests pass
5. ✅ Training runs faster (measured)

---

## Task Breakdown

### Task 1.1: Add Release Profile to Cargo.toml

**Description**: Configure optimized release profile settings.

**Implementation**:
```toml
# Add to Cargo.toml

[profile.release]
lto = "fat"           # Full link-time optimization across all crates
codegen-units = 1     # Better optimization (single codegen unit)
panic = "abort"       # No unwinding overhead, smaller binary
opt-level = 3         # Maximum optimization level

[profile.release.build-override]
opt-level = 3         # Optimize build scripts too
```

**Rationale**:
- `lto = "fat"`: Enables cross-crate inlining. Critical for small functions like `execute_soft()` called millions of times
- `codegen-units = 1`: LLVM can optimize globally, not per-unit. Slower compile, faster runtime
- `panic = "abort"`: Removes unwinding machinery, slightly faster
- `opt-level = 3`: Maximum optimization (aggressive inlining, vectorization)

**Tests**:
```bash
# Verify build succeeds
cargo build --release

# Verify tests pass
cargo test --lib --release
```

**Exit Criteria**:
- [ ] Cargo.toml contains `[profile.release]` section
- [ ] `cargo build --release` succeeds
- [ ] `cargo test --lib --release` passes (all 143+ tests)

---

### Task 1.2: Document RUSTFLAGS for Native CPU

**Description**: Add documentation for running with native CPU optimizations.

**Implementation**:

Update `README.md` build section (or create if needed):
```markdown
### Optimized Build

For maximum performance on your specific CPU:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This enables AVX2/AVX-512 vectorization if your CPU supports it.

To make this permanent, create `.cargo/config.toml`:
```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```
```

**Note**: We document but don't commit `.cargo/config.toml` since it's machine-specific.

**Exit Criteria**:
- [ ] README.md documents `RUSTFLAGS="-C target-cpu=native"`
- [ ] Instructions are clear for users

---

### Task 1.3: Benchmark Before/After

**Description**: Measure actual speedup from profile changes.

**Implementation**:

Create a simple benchmark script or document the benchmark process:

```bash
# Before changes (record time)
time cargo run --bin train_checkerboard --release -- --small --epochs=50

# After changes (compare)
RUSTFLAGS="-C target-cpu=native" time cargo run --bin train_checkerboard --release -- --small --epochs=50
```

Expected output format:
```
Epoch   50: soft_loss=XX.XX, hard_loss=XX.XX, acc=XX.XX% [XX.Xs]
```

**Measurement Points**:
1. Total wall-clock time for 50 epochs
2. Per-epoch time (from log output)

**Exit Criteria**:
- [ ] Baseline time recorded (before changes)
- [ ] Optimized time recorded (after changes)
- [ ] Speedup calculated and documented

---

### Task 1.4: Verify No Regression

**Description**: Ensure optimization flags don't change numerical behavior.

**Implementation**:

Run training with both profiles and verify:
1. Same soft loss values (within floating-point tolerance)
2. Same hard accuracy values
3. No crashes or panics

```bash
# Compare outputs
cargo run --bin train_checkerboard --release -- --small --epochs=10 2>&1 | tee before.log
RUSTFLAGS="-C target-cpu=native" cargo run --bin train_checkerboard --release -- --small --epochs=10 2>&1 | tee after.log

# Diff should show similar loss/accuracy values
diff before.log after.log
```

**Exit Criteria**:
- [ ] Loss values are similar (within 1% relative difference)
- [ ] No new warnings or errors
- [ ] Training behaves identically

---

## Final Checklist

| Task | Status | Verified By |
|------|--------|-------------|
| 1.1 Release profile added | ⬜ | `cargo build --release` succeeds |
| 1.2 RUSTFLAGS documented | ⬜ | README.md updated |
| 1.3 Benchmark completed | ⬜ | Speedup measured |
| 1.4 No regression | ⬜ | Same numerical results |

---

## Implementation Notes

### Why These Specific Settings

| Setting | Default | Our Value | Why |
|---------|---------|-----------|-----|
| `lto` | `false` | `"fat"` | Cross-crate inlining critical for small gate functions |
| `codegen-units` | 16 | 1 | Global optimization view |
| `panic` | `"unwind"` | `"abort"` | Training doesn't need graceful panic handling |
| `opt-level` | 3 | 3 | Already default for release, explicit is clearer |

### Compile Time Trade-off

LTO + codegen-units=1 increases compile time significantly:
- Before: ~30 seconds
- After: ~2-3 minutes

This is acceptable because:
1. We only rebuild when code changes
2. Training runs are hours long
3. Runtime speedup is permanent

### Alternative: Profile-Guided Optimization (PGO)

For even more speed (additional 10-20%), PGO could be used:
```bash
# Generate profile
RUSTFLAGS="-Cprofile-generate=/tmp/pgo" cargo run --release -- --epochs=10

# Build with profile
RUSTFLAGS="-Cprofile-use=/tmp/pgo/merged.profdata" cargo build --release
```

This is more complex and deferred to future optimization.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Longer compile times | Certain | Low | Only affects rebuilds, acceptable trade-off |
| Platform-specific issues | Low | Low | LTO is well-tested in Rust ecosystem |
| Numerical differences | Very Low | Medium | Test shows same results |

---

## Rollback Plan

If issues occur, simply remove the `[profile.release]` section from Cargo.toml. No code changes are made in this phase.

---

## Next Phase

After Phase 1 is complete, proceed to **Performance Phase 2: Parallelize Backward Pass** for the largest single speedup opportunity.
