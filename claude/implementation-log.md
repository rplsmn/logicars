# Logicars Implementation Log

This document tracks the implementation progress, key learnings, and the successful workflow pattern for building the differentiable logic CA system.

---

## Development Workflow (PROVEN PATTERN)

1. **Create Todo List**: Use TodoWrite to break down the phase into specific tasks
2. **Write Unit Tests First**: Create tests for core functionality before implementation
3. **Implement Core Logic**: Write the actual implementation
4. **Run Tests Continuously**: Test after each major component
5. **Build & Run Integration Tests**: Create binaries to test full workflows
6. **Verify Exit Criteria**: Ensure phase goals are met
7. **Document & Commit**: Update this log, commit with detailed message
8. **Push PR**: Push to branch and create pull request

---

## Phase 0: Foundation ✅ COMPLETE

All foundation phases completed successfully. Core primitives validated and working.

| Phase | Description | Gates | Tests | Key Result |
|-------|-------------|-------|-------|------------|
| 0.1 | Single Gate Training | 1 | 10 | 100% accuracy on AND/OR/XOR |
| 0.2 | Gate Layer | N | 17 | 8 gates learning 8 ops simultaneously |
| 0.3 | Multi-Layer Circuits | 2-3 layers | 25 | XOR via 2-layer circuit, backprop verified |

### Key Technical Decisions (Phase 0)

- **Truth Table Encoding**: Enum values encode truth tables (e.g., AND=8=0b1000), execute via `(value >> bits) & 1`
- **Soft Execution**: Boolean inputs as probabilities in [0,1], AND(a,b)=a*b, OR(a,b)=a+b-a*b
- **Gradient Verification**: Numerical gradient checking validates all analytical gradients
- **Pass-through Init**: Gate index 3 initialized to logit=10.0 for training stability
- **Hyperparameters**: LR=0.05, gradient clipping=100.0, AdamW with weight_decay=0.01

### Key Learnings (Phase 0)

1. **XOR takes ~2x longer** than AND/OR to converge (non-linearly separable)
2. **Networks find shortcuts**: Multi-layer XOR often learns XOR directly in layer 1, uses pass-through in later layers
3. **Convergence time is constant** with layer size (3-8 gates: ~1000 iterations each)
4. **Test-first development** catches bugs early; numerical gradient checking is essential

### Code Organization (Phase 0)

```
src/
├── phase_0_1.rs       # BinaryOp enum, ProbabilisticGate
├── phase_0_2.rs       # GateLayer, LayerTruthTable, LayerTrainer
├── phase_0_3.rs       # Circuit, ConnectionPattern, CircuitTrainer
├── optimizer.rs        # AdamW implementation
├── trainer.rs          # Single gate trainer
└── bin/
    ├── train_gate.rs   # Phase 0.1 demo
    ├── train_layer.rs  # Phase 0.2 demo
    └── train_circuit.rs # Phase 0.3 demo
```

---

## QA Analysis: Architecture Deviation ⚠️ CRITICAL

**Date**: 2026-01-01
**Reviewer**: Claude (QA session)

### Problem Identified

The current Phase 1.1 implementation has a **fundamental architectural mismatch** with the reference implementation, explaining the 81% accuracy ceiling.

### Reference Architecture (Game of Life)

```
┌─────────────────────────────────────────────────────────────┐
│ PERCEPTION MODULE                                            │
│ - 16 parallel kernels                                        │
│ - Each kernel: [9→8→4→2→1] = 15 gates                       │
│ - First layer: center vs each of 8 neighbors                │
│ - Output: 16 feature bits                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
              [center_cell, k1, k2, ..., k16] = 17 bits
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ UPDATE MODULE                                                │
│ - 23 layers: [17→128]×16, then [64→32→16→8→4→2→1]          │
│ - Connection topology: "unique" throughout                   │
│ - Output: 1 bit (next cell state)                           │
└─────────────────────────────────────────────────────────────┘

TOTAL: 336 active gates
```

### Current Rust Implementation

```
┌─────────────────────────────────────────────────────────────┐
│ DeepPerceptionKernel (single, doing everything)             │
│ - 1 kernel (not 16)                                         │
│ - Layers: [32→16→8→4→2→1] = 66 gates                       │
│ - Directly predicts GoL (no update module)                  │
│ - Center cell mixed into network                            │
└─────────────────────────────────────────────────────────────┘

TOTAL: 66 gates (vs 336 reference)
```

### Key Differences

| Aspect | Reference | Current Rust |
|--------|-----------|--------------|
| Perception kernels | 16 parallel | 1 single |
| Gates per kernel | ~15 | 66 (doing both jobs) |
| Update module | 23 layers, 128+ gates | **MISSING** |
| Total gates | 336 | 66 |
| Center cell | Explicitly preserved & concatenated | Mixed into perception |
| Architecture | Perception→features, Update→decision | Single network→decision |

### Why 81% is the Ceiling

1. **Insufficient capacity**: 66 gates vs 336 (5× fewer)
2. **Missing separation of concerns**: Perception extracts features, update makes decisions
3. **Center cell not preserved**: GoL rules depend on both center state AND neighbor count
4. **No multi-kernel diversity**: 16 kernels learn different features; provides rich information to update module

### Recommended Fix

1. **Phase 1.1 (perception)**: Implement 4-16 **parallel** kernels, each outputting 1 feature bit
2. **Phase 1.2 (update)**: Implement update module taking `[center, k1, k2, ..., k16]` → next state
3. **Increase capacity**: Update module needs 16+ layers with 128+ gates per layer

### Reference Implementation

A snapshot of the Python/JAX reference code is now available in `reference/difflogic_ca.py` for direct comparison.

---

## Phase 1.1: Perception Circuit - IN PROGRESS

**Date**: 2026-01-01
**Status**: EXIT CRITERIA NOT MET (81% accuracy, need 95%)
**Branch**: `claude/review-docs-update-claude-Xf4tl`

### What Was Implemented

1. **Grid** (`src/phase_1_1.rs`): 2D binary grid with toroidal boundaries
2. **Neighborhood**: 3x3 extraction, 9-bit indexing, GoL next-state computation
3. **GolTruthTable**: All 512 configurations with precomputed targets
4. **PerceptionKernel**: 3-layer shallow kernel (~25 gates)
5. **DeepPerceptionKernel**: 6-layer deep kernel (66 gates)
6. **Trainers**: PerceptionTrainer, DeepPerceptionTrainer with backprop

### Test Results

- **35 unit tests passing** (gradient checking verified)
- **Shallow kernel**: 73.44% accuracy (plateaued)
- **Deep kernel**: 80.86% accuracy (best achieved: 81.84%)

### Exit Criteria Status

- ❌ >95% accuracy on 512 configurations (achieved 80.86%)
- ✅ Training infrastructure works correctly
- ✅ Gradients verified numerically

### Next Steps for Phase 1.1

Based on QA analysis, the architecture needs correction:

1. **Option A**: Restructure Phase 1.1 to output features (not predictions)
   - Change to 4-16 parallel kernels
   - Each kernel outputs 1 bit
   - Defer decision-making to Phase 1.2 (update module)

2. **Option B**: Massively increase capacity (not recommended)
   - Would require 300+ gates in single network
   - Still missing the right inductive bias

**Recommended**: Option A - Follow reference architecture

### Commands

```bash
# Run all unit tests (35 tests)
cargo test --lib

# Run Phase 1.1 tests
cargo test phase_1_1 --lib

# Run perception training (takes ~5-10 minutes)
cargo run --bin train_perception --release
```

---

## Development Environment

- **Rust**: 1.91.1
- **Platform**: Linux 4.4.0
- ✅ Native testing works (no Docker needed)

---

## Workflow Checklist

Use for each phase:

- [ ] Read `claude/plan.md` for phase requirements
- [ ] Read this log for context and learnings
- [ ] Create TodoWrite list with specific tasks
- [ ] Write unit tests first
- [ ] Implement core logic
- [ ] Run `cargo test --lib` continuously
- [ ] Create integration test binary if needed
- [ ] Verify all exit criteria met
- [ ] Update this log
- [ ] Commit with detailed message
- [ ] Push to branch, create PR

---

## Common Patterns

### Numerical Gradient Checking

```rust
let epsilon = 1e-5;
for i in 0..num_params {
    params[i] += epsilon;
    let loss_plus = compute_loss();
    params[i] -= 2.0 * epsilon;
    let loss_minus = compute_loss();
    params[i] += epsilon;  // restore

    let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
    assert_relative_eq!(analytical[i], numerical, epsilon=1e-4);
}
```

### Training Pattern

```rust
let truth_table = TruthTable::for_operation(BinaryOp::And);
let mut trainer = GateTrainer::new(0.05);
let result = trainer.train(&truth_table, max_iters, target_loss, verbose);
assert!(result.meets_exit_criteria(BinaryOp::And));
```

---

## Questions for Later Phases

1. What's the optimal layer width for update module? (Reference: 128-512)
2. Should we add learning rate scheduling?
3. How to visualize gate probability evolution?
4. Can we parallelize gate training further?

---

---

## QA Review: Scope Correction (2026-01-02)

**Critical finding**: Project had tunnel vision on GoL when the paper's value is in multi-bit experiments.

**Changes made**:
- `claude/plan.md` restructured: Multi-state moved from Phase 4.2 → Phase 1.1
- `reference/difflogic_ca.py` updated: Added all 5 experiment hyperparams
- `reference/README.md` rewritten: Full documentation of all experiments
- Created `claude/qa-review-1.md`: Detailed findings and rationale

See `claude/qa-review-1.md` for full details.

---

## Phase 1.1: N-bit Grid Implementation ✅ COMPLETE

**Date**: 2026-01-02
**Status**: ALL EXIT CRITERIA MET

### What Was Implemented

Created `src/grid.rs` with N-bit capable grid and neighborhood types:

1. **`NGrid`** - Multi-channel grid with runtime channel count (1-128)
   - Supports both periodic (toroidal) and non-periodic (clamped) boundaries
   - Soft (f64) representation for training, with `to_hard()` for inference
   - Channel-aware storage: `Vec<f64>` with length = width × height × channels
   - Full compatibility helpers for GoL (C=1): `from_bool_grid()`, `to_bool_grid()`

2. **`NNeighborhood`** - 3×3 neighborhood extraction with 9C values
   - Flat storage: [NW_ch0..NW_chC, N_ch0..N_chC, ..., SE_ch0..SE_chC]
   - Position-based and channel-based accessors
   - GoL helpers: `from_gol_index()`, `to_gol_index()`, `gol_next_state()`

3. **`BoundaryCondition`** enum - Periodic vs NonPeriodic

### Test Results

```
running 60 tests (25 new for grid module)
test result: ok. 60 passed; 0 failed
```

New tests cover:
- C=1: 8 tests (GoL validation, wrapping, indexing)
- C=8: 4 tests (Checkerboard, multi-channel access, non-periodic)
- C=64: 3 tests (Colored G)
- C=128: 4 tests (Growing Lizard, large grids)
- Edge cases: 6 tests (minimal grid, 2×2 wrapping, invalid channels)

### Exit Criteria: ✅ ALL MET

- ✅ Unit tests for C=1, C=8, C=64, C=128
- ✅ Neighborhood extraction works for all channel counts
- ✅ Periodic and non-periodic boundaries implemented
- ✅ Backward compatibility with GoL (C=1) via helper methods
- ✅ All 60 tests pass

### Key Technical Decisions

1. **Dynamic channels (runtime)** over const generics for flexibility across experiments
2. **Flat Vec<f64> storage** for efficient memory layout and easy indexing
3. **Separate module** (`grid.rs`) rather than modifying `phase_1_1.rs` - old Grid coexists for transition
4. **Soft-first representation** - store f64, convert to hard/bool as needed

### Code Organization

```
src/
├── grid.rs              # NEW: NGrid, NNeighborhood, BoundaryCondition
├── phase_1_1.rs         # OLD: Grid (Vec<bool>), Neighborhood, GoL training
└── lib.rs               # Updated exports
```

### Commands for Next Developer

```bash
# Run all tests
cargo test --lib

# Run only grid tests
cargo test grid --lib

# Verify specific channel counts
cargo test c128 --lib
cargo test c64 --lib
```

---

## Next Steps

**Phase 1.2: Perception Module** - Implement parallel perception kernels using NGrid:
- 4-16 parallel kernels, each outputting feature bits
- Use `NNeighborhood` as input (9C values)
- Output: concatenation of center cell + kernel outputs
- Match reference architecture exactly

---

**Last Updated**: 2026-01-02
**Current Phase**: 1.1 ✅ COMPLETE → Ready for 1.2 (Perception Module)
**Status**: N-bit Grid implemented with 25 tests passing
