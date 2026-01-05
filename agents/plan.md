# Logicars Development Roadmap

## Quick Reference

| Phase | Status | Description |
|-------|--------|-------------|
| 0.x | ✅ | Gate primitives, layers, circuits |
| 1.x | ✅ | N-bit architecture, GoL validation (99.41%) |
| 2.1 | 🚧 | Checkerboard C=8 sync |
| 2.2 | ⬜ | Checkerboard C=8 async |
| 2.3 | ⬜ | Growing Lizard C=128 |
| 2.4 | ⬜ | Colored G C=64 |
| 3.x | ⬜ | Library API, serialization |

**Current Focus**: Phase 2.1 - Checkerboard training

---

## Primary References

1. **Paper**: [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/)
2. **Reference Implementation**: `reference/difflogic_ca.py`
3. **Full Documentation**: `reference/README.md`

---

## Design Principle: N-bit From Start

Design for N-bit cells from Phase 1. GoL is just N=1.
- `NGrid` supports C=1-128 channels
- Perception and Update modules are channel-aware
- No refactoring needed for multi-bit experiments

---

## Completed Phases (Summary)

### Phase 0: Foundation ✅
Gate primitives with backprop verified. See `phase_0_*.rs`.

### Phase 1: N-bit Core ✅
- **1.1-1.4**: NGrid, Perception, Update, Training modules
- **1.5**: GoL 99.41% accuracy, blinker/glider work
- Architecture matches reference exactly

---

## Phase 2: Multi-bit Experiments

**Goal**: Validate N-bit architecture on progressively harder tasks

### 2.1 Checkerboard (C=8) 🚧 IN PROGRESS

First multi-channel test. Validates:
- 8-bit state handling
- Non-periodic boundaries
- Multi-step rollout (20 steps)

**Implementation status**:
- ✅ Model architecture: perception (16 kernels, [9→8→4→2]) + update ([264→256×10→...→8])
- ✅ Training binary: `src/bin/train_checkerboard.rs`
- ✅ Unit tests: 14 tests passing
- ⬜ Training: Requires long training run (hours)

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
- `DiffLogicCA::new(config)` - create model ✅
- `model.step(grid)` - single step ✅
- `model.simulate(grid, steps)` - multi-step ✅

### 3.2 Serialization ✅ IMPLEMENTED
- `HardCircuit::from_soft(model)` - export trained model
- `circuit.save(path)` / `HardCircuit::load(path)` - JSON persistence
- `circuit.active_gate_count()` - count non-pass-through gates
- `circuit.gate_distribution()` - analyze gate types

### 3.3 Testing & Docs
- Comprehensive test suite (127 tests)
- Tutorial: "Train your first CA" ⬜
- Benchmark suite vs reference impl ⬜

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

1. **N-bit from start**: No refactoring for multi-channel ✅
2. **Verification first**: Never proceed with failing tests ✅
3. **Match reference**: Compare outputs layer-by-layer ✅
4. **GoL is validation, not goal**: Real value is Phases 2.x

---

## Key Implementation Tricks

| Trick | Value | Notes |
|-------|-------|-------|
| Pass-through init | logit=10.0 | Gate index 3 (A) |
| Gradient clipping | 100.0 | Prevents explosion |
| AdamW β2 | 0.99 | Not 0.999 - escapes local minima |
| Fire rate | 0.6 | Async training |
| Soft/Hard | softmax/argmax | Train soft, eval hard |

**Connection Types**:
- `first_kernel`: center vs 8 neighbors (perception layer 1)
- `unique`: unique pair connections (all other layers)

---

## Learnings & Risk Mitigation

1. **Architecture mismatch** → Count gates, verify perception+update separation
2. **81% ceiling** → Was wrong architecture; center cell must concat, not mix
3. **β2=0.99** → Escapes local minima faster than 0.999
4. **Per-example training slow** → Batching would help (future optimization)
5. **Test incrementally** → C=1 → C=8 → C=64 → C=128
