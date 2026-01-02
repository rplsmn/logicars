# Logicars Development Roadmap

## Primary References

1. **Paper**: [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/)
2. **Reference Implementation**: See `reference/difflogic_ca.py` - JAX/Python with ALL experiment hyperparameters
3. **Full Documentation**: See `reference/README.md` - all 5 experiments documented

---

## Design Principle: N-bit From Start

**QA Review Finding** (2026-01-01): The original plan had multi-state support at Phase 4.2, creating architectural debt. The paper's real value is in multi-bit experiments (Checkerboard, Lizard, Colored G), not just GoL.

**New Principle**: Design for N-bit cells from Phase 1. GoL is just N=1.

This means:
- `Grid<C>` where C = channels (1-128 bits per cell)
- Perception and Update modules are channel-aware
- GoL validates the architecture with simplest case (C=1)
- No refactoring needed when moving to multi-bit experiments

---

## Phase 0: Foundation & Verification ✅ COMPLETE

**Goal**: Rock-solid single-gate primitives

| Phase | Status | Description |
|-------|--------|-------------|
| 0.1 | ✅ | Single gate training (AND/OR/XOR) |
| 0.2 | ✅ | Gate layer (8 ops simultaneously) |
| 0.3 | ✅ | Multi-layer circuits (XOR via 2-layer) |

**Key Learnings**: See `claude/implementation-log.md`

---

## Phase 1: N-bit Core Architecture

**Goal**: Perception + Update modules that work for ANY channel count

### 1.1 N-bit Grid and Neighborhood ✅ COMPLETE

**What to build**:
```rust
// Option A: Const generic (compile-time channels)
struct Grid<const C: usize> {
    width: usize,
    height: usize,
    cells: Vec<[f64; C]>,
}

// Option B: Dynamic (runtime channels) - more flexible
struct Grid {
    width: usize,
    height: usize,
    channels: usize,
    cells: Vec<f64>,  // Flat: [cell0_ch0, cell0_ch1, ..., cell1_ch0, ...]
}
```

**Requirements**:
- Extract 3×3 neighborhood (9 cells × C channels = 9C values)
- Support periodic and non-periodic boundaries
- Convert to/from soft (f64) and hard (bool) representations

**Exit criteria**:
- Unit tests for C=1, C=8, C=64, C=128
- Neighborhood extraction matches reference impl

### 1.2 Perception Module (Parallel Kernels) ✅ COMPLETE

**Architecture** (from reference):
```
Input: 9 cells × C channels
       ↓
K parallel kernels, each:
  Layer 1: 8 gates (first_kernel topology)
  Layers 2+: unique connections
  Output: 1+ bits per kernel
       ↓
Output: [center_channels, kernel_1_out, ..., kernel_K_out]
```

**Key decisions**:
- K = 4-16 kernels (configurable)
- Each kernel is independent (no weight sharing)
- Center cell is preserved and concatenated with outputs

**Exit criteria**:
- Forward pass matches reference impl
- Gradients verified numerically
- Works for C=1 (GoL config: 16 kernels, [9→8→4→2→1])

### 1.3 Update Module ✅ COMPLETE

**Architecture**:
```
Input: center (C bits) + kernel outputs
       ↓
Deep network: [input_size, 128-512, ..., C]
All "unique" connections
       ↓
Output: C bits (next cell state)
```

**Exit criteria**:
- Forward pass matches reference
- Backprop through full perception→update chain
- Works for GoL config: [17→128×10→64→...→1]

### 1.4 Training Loop (MSE Loss) ⬅️ CURRENT

**Requirements**:
- `loss = sum((predicted - target)²)` per cell per channel
- AdamW with gradient clipping (100.0)
- Support sync and async training modes
- Fire rate masking for async (0.6)

**Exit criteria**:
- Loss decreases on random data
- Matches reference loss computation

### 1.5 GoL Validation (C=1)

**This is the real test**: Does the N-bit architecture work for the simplest case?

**Training setup**:
- C=1 (single bit per cell)
- 16 perception kernels
- 18-layer update module
- All 512 neighborhood configurations

**Exit criteria**:
- >95% hard accuracy on GoL (the 81% ceiling is broken)
- Architecture matches reference exactly
- Gliders and blinkers work in simulation

---

## Phase 2: Multi-bit Experiments

**Goal**: Validate N-bit architecture on progressively harder tasks

### 2.1 Checkerboard (C=8)

First multi-channel test. Validates:
- 8-bit state handling
- Non-periodic boundaries
- Multi-step rollout (20 steps)

**Exit criteria**:
- Pattern emerges from seed
- Generalizes to larger grids (16×16 → 64×64)

### 2.2 Checkerboard Async (C=8)

Same as 2.1 but with async training:
- Fire rate masking
- Self-healing behavior

**Exit criteria**:
- Fault tolerance demonstrated
- Pattern recovers from damage

### 2.3 Growing Lizard (C=128)

Complex pattern generation:
- 128-bit state (largest)
- 12 growth steps
- Fewer kernels (4 vs 16)

**Exit criteria**:
- Lizard pattern grows from seed
- Works on 40×40 (trained on 20×20)

### 2.4 Colored G (C=64)

Most complex circuit:
- 64-bit state (RGB)
- 927 active gates
- 15 generation steps

**Exit criteria**:
- Colored G pattern generated
- 8-color palette visible

---

## Phase 3: Library API

**Goal**: Clean abstractions for users

### 3.1 API Design
- `DiffLogicCA::new(config)` - create model
- `model.train(data, epochs)` - training
- `model.step(grid)` - single step
- `model.simulate(grid, steps)` - multi-step

### 3.2 Serialization
- Save/load trained circuits
- Export gate weights and connections
- Hard-decode to pure logic circuit

### 3.3 Testing & Docs
- Comprehensive test suite
- Tutorial: "Train your first CA"
- Benchmark suite vs reference impl

---

## Phase 4: Advanced Features

### 4.1 Architecture Search
- Auto-tune kernel count, layer depth
- Find minimal circuit for accuracy

### 4.2 Larger Neighborhoods
- 5×5, 7×7 (currently only 3×3)

### 4.3 Inverse Problems
- Given behavior → find rule
- Self-healing optimization

---

## Phase 5: Ecosystem

### 5.1 Python Bindings (PyO3)
### 5.2 Visualization Tools
### 5.3 WASM Demo

---

## Critical Success Factors

1. **N-bit from start**: No refactoring for multi-channel
2. **Verification first**: Never proceed with failing tests
3. **Match reference**: Compare outputs layer-by-layer
4. **GoL is validation, not goal**: The real value is Phases 2.x

---

## Risk Mitigation

### Architecture Mismatch (LEARNED)
Previous 81% ceiling was due to wrong architecture. Always:
- Count gates and compare to reference
- Verify perception + update separation
- Check center cell concatenation

### Multi-bit Complexity
Test incrementally: C=1 → C=8 → C=64 → C=128

### Over-engineering
Keep it simple. N-bit is the ONE complexity we're adding early because it's foundational.

---

## Key Implementation Tricks (from Reference)

1. **Pass-through init**: Gate index 3 = 10.0 logit
2. **Gradient clipping**: 100.0
3. **Soft/hard**: Train soft (softmax), eval hard (argmax)
4. **Center concat**: Always preserve center cell
5. **Fire rate**: 0.6 for async training
6. **first_kernel**: Layer 1 perception topology
7. **unique**: All other layer connections

---

## What Changed from Original Plan

| Aspect | Before | After |
|--------|--------|-------|
| Multi-state | Phase 4.2 | Phase 1.1 (foundational) |
| GoL | Phase 1 goal | Phase 1.5 (validation of N-bit) |
| Checkerboard | Not planned | Phase 2.1-2.2 |
| Lizard/Colored G | Not planned | Phase 2.3-2.4 |
| Grid type | `Vec<bool>` | `Grid<C>` (N-bit) |

**Rationale**: See `claude/qa-review-1.md`
