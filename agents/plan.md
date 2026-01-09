# Logicars Development Roadmap

## Project Overview

Implementation of differentiable logic cellular automata for learning CA rules, based on the paper [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/).

**Core Achievement**: N-bit architecture from start - no refactoring needed for multi-channel experiments.

---

## Primary References

1. **Paper**: [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/)
2. **Reference Implementation**: `reference/diffLogic_CA.ipynb` (ground truth)
3. **Compacted Python**: `reference/difflogic_ca.py` (may have errors - trust notebook)

**Golden Rule**: ALWAYS TRUST THE NOTEBOOK - it's tested for replication.

---

## Phase Status Overview

| Phase | Status | Description | Completion Date |
|-------|--------|-------------|-----------------|
| 0.x | ✅ | Gate primitives, backprop verified | 2026-01-05 |
| 1.x | ✅ | N-bit architecture, GoL 99.41% | 2026-01-06 |
| 2.1 | ✅ | Checkerboard C=8 sync training | 2026-01-09 |
| 2.1a | 🚧 | Checkerboard analysis & docs | In progress |
| 2.2 | ⬜ | Checkerboard async (self-healing) | Not started |
| 2.3 | ⬜ | Growing Lizard C=128 | Not started |
| 2.5 | ⬜ | Minimal PyO3 bindings | Not started |
| 2.4 | ⬜ | Colored G C=64 | Not started |
| 3.x | 🚧 | Library API (partial) | Ongoing |
| 4.x | 🚧 | Performance (partial) | Ongoing |
| 5.x | ⬜ | Ecosystem (PyO3, extendr, WASM) | Not started |

**Current Focus**: Phase 2.1a - Complete checkerboard analysis and documentation

---

## Completed Phases

### Phase 0: Foundation ✅

**Goal**: Implement and verify differentiable logic gates with backpropagation.

**What was built**:
- 16 binary operations (AND, OR, XOR, NAND, etc.)
- Probabilistic gates with soft/hard modes
- Gate layers with backpropagation
- Circuit composition and training

**Key files**: `src/phase_0_1.rs`, `src/phase_0_2.rs`, `src/phase_0_3.rs`

**Exit criteria met**:
- ✅ All 16 binary ops implemented
- ✅ Soft mode: softmax-weighted average (differentiable)
- ✅ Hard mode: argmax selection (discrete)
- ✅ Backpropagation verified with gradient checks
- ✅ Pass-through initialization (logit=10.0) works

---

### Phase 1: N-bit Core Architecture ✅

**Goal**: Build complete N-channel CA architecture that works for any channel count (C=1 to C=128).

**What was built**:
- `NGrid`: Multi-channel grid with boundary conditions
- `PerceptionModule`: Multi-kernel perception with configurable layers
- `UpdateModule`: Deep network for state updates
- `DiffLogicCA`: Complete trainable CA model
- `TrainingLoop`: BPTT with AdamW optimizer

**Validation**: Game of Life (C=1)
- Training: 99.41% hard accuracy
- Blinker and glider patterns work correctly
- Model: 1647 gates total (240 perception + 1407 update)

**Key files**: `src/grid.rs`, `src/perception.rs`, `src/update.rs`, `src/training.rs`

**Exit criteria met**:
- ✅ Architecture supports C=1 to C=128 channels
- ✅ Perception extracts features from 3×3 neighborhoods
- ✅ Update network decides state transitions
- ✅ BPTT training with gradient clipping (100.0)
- ✅ Batch training support
- ✅ GoL validation: 99.41% accuracy
- ✅ 148 tests passing

---

### Phase 2.1: Checkerboard Synchronous (C=8) ✅

**Goal**: First multi-channel experiment - prove architecture generalizes beyond GoL.

**What was built**:
- Checkerboard pattern generation (2×2 squares, 16×16 grid)
- Multi-channel loss (channel 0 only, others are working memory)
- Full model: 3040 gates (16 kernels, 16-layer update)
- Small model: 1496 gates (16 kernels, 9-layer update)
- Training binary with logging and checkpointing

**Training results** (2026-01-09):
- Epoch 512: First 100% hard accuracy
- Epochs 512-637: Maintained 100% accuracy, hard_loss=0.0
- Total time: ~35 minutes (2121 seconds)
- Matches reference implementation performance

**Key files**: `src/checkerboard.rs`, `src/bin/train_checkerboard.rs`

**Exit criteria met**:
- ✅ Pattern emerges from random seed in 20 steps
- ✅ Training converges to 100% accuracy
- ✅ Soft and hard losses both converge
- ⬜ Gate distribution analysis (Phase 2.1a)
- ⬜ Generalization to 64×64 grid (Phase 2.1a)
- ⬜ Visual reconstruction GIF (Phase 2.1a)

**Critical learnings**:
- Loss must be channel-specific (channel 0 only for checkerboard)
- Gradient scaling: raw sum, no averaging (scale=1.0)
- 200-epoch plateau is expected (gates escaping pass-through initialization)
- Batch training (batch_size=2) helps convergence

---

## Active Phase

### Phase 2.1a: Checkerboard Analysis & Documentation 🚧

**Goal**: Complete scientific analysis and documentation before moving to Phase 2.2.

**Tasks**:

1. **Gate Distribution Analysis**
   - Convert trained model to HardCircuit
   - Count active vs pass-through gates
   - Analyze which operations were learned
   - Export analysis to text/CSV

2. **Visual Reconstruction**
   - Generate animated GIF of 20-step rollout
   - Show pattern emergence from random seed
   - Throwaway code in binary is fine

3. **Generalization Testing**
   - Test trained 16×16 model on 64×64 grid
   - Success criteria: >95% accuracy
   - May need more than 20 steps for larger grids

4. **Documentation**
   - Update implementation-log.md with results
   - Mark Phase 2.1 as fully complete
   - Capture key insights for next phases

**Exit criteria**:
- ✅ Gate distribution analysis saved
- ✅ Animated GIF generated
- ✅ 64×64 generalization validated (>95% accuracy)
- ✅ All results documented in implementation-log.md
- ✅ Phase 2.1 marked ✅ COMPLETE in plan.md

**Estimated effort**: 8-10 hours

**Implementation details**: See `plans/phase-2.1a-analysis.md` (to be created)

---

## Planned Phases

### Phase 2.2: Checkerboard Async (C=8) ⬜

**Goal**: Add asynchronous training with fire rate masking - demonstrate self-healing.

**What to build**:
- Fire rate masking (fire_rate=0.6)
- Fault tolerance visualization
- Damage recovery tests

**Exit criteria**:
- Pattern emerges with async updates
- Model recovers from random damage
- Self-healing demonstrated visually

**Visualization**: Rust throwaway code (GIF of damage recovery)

---

### Phase 2.3: Growing Lizard (C=128) ⬜

**Goal**: Test highest channel count with complex morphogenesis pattern.

**What to build**:
- Lizard pattern generation
- Growth simulation (12 steps)
- Fewer kernels (4 vs 16) to match reference

**Exit criteria**:
- Lizard pattern grows from seed
- Generalizes to 40×40 (trained on 20×20)
- Visual confirmation of growth stages

**Visualization**: Rust throwaway code (GIF of growth sequence)

**After Phase 2.3**: Stop and build PyO3 bindings before Colored G

---

### Phase 2.5: Minimal PyO3 Bindings ⬜

**Goal**: Build minimal Python API for development of remaining experiments.

**Rationale**: Colored G (Phase 2.4) needs proper 8-color visualization. Python's matplotlib is far superior to Rust for this. By this point, core architecture is stable and we know what API we need.

**What to build**:
- `logicars-py` crate with PyO3 bindings
- Core API: create model, train, step, save/load
- NumPy array integration for grids
- Basic examples and tests

**API scope**:
```python
import logicars

# Create and train model
model = logicars.DiffLogicCA(channels=8, kernels=16, ...)
model.train(train_data, config)

# Inference
output = model.step(input_grid)

# Serialization
model.save("model.json")
model2 = logicars.load("model.json")

# Analysis
print(model.gate_distribution())
```

**Exit criteria**:
- ✅ Core training loop works from Python
- ✅ Grid I/O via NumPy arrays
- ✅ Model save/load works
- ✅ Can reproduce checkerboard training from Python
- ✅ Basic documentation and examples

**Estimated effort**: 1-2 weeks

**Implementation details**: See `plans/phase-2.5-pyo3.md` (to be created)

---

### Phase 2.4: Colored G (C=64) ⬜

**Goal**: Most complex pattern - 8-color palette visualization.

**What to build**:
- Colored G pattern (8 colors, C=64)
- 927 active gates (reference metric)
- 15 generation steps

**Development environment**: Python (Jupyter notebooks preferred)

**Exit criteria**:
- Colored G pattern generated
- 8-color palette clearly visible
- Gate count matches reference (~927 active)

**Visualization**: Python (matplotlib, proper color mapping)

---

### Phase 3: Library API 🚧

**Status**: Partially complete

**Completed** (3.2):
- ✅ `HardCircuit::from_soft()` - export trained models
- ✅ `circuit.save(path)` / `HardCircuit::load(path)` - JSON serialization
- ✅ `circuit.active_gate_count()` - count non-pass-through gates
- ✅ `circuit.gate_distribution()` - analyze gate types
- ✅ Comprehensive test suite (148 tests)

**Remaining** (3.3):
- ⬜ Tutorial: "Train your first CA"
- ⬜ Benchmark suite vs reference implementation
- ⬜ API documentation (rustdoc)

---

### Phase 4: Performance Optimization 🚧

**Status**: Partially complete

**Completed optimizations**:
- ✅ Phase 1: Release profile tuning (LTO, codegen-units=1)
- ✅ Phase 2: Parallel backward pass (rayon)
- ✅ Phase 3: Batch parallelization (rayon)
- ✅ Phase 5: Softmax probability caching (RwLock)

**Speedup achieved**: ~10-15x faster than initial implementation

**Remaining optimizations**:
- ⬜ Phase 4: f32 precision (documented, not implemented)
- ⬜ SIMD gate operations (future work)
- ⬜ Memory pooling (future work)

**Performance files**: See `plans/perf-*.md` for detailed phase plans

---

### Phase 5: Ecosystem ⬜

**Planned components**:

**5.1: Production PyO3 API**
- Refine Phase 2.5 bindings into production API
- Advanced features, full documentation
- PyPI package

**5.2: extendr R Bindings**
- R package using extendr framework
- Same core API as Python
- CRAN package

**5.3: Visualization Tools**
- Standalone visualization utilities
- Interactive demos

**5.4: WASM Demo**
- Browser-based CA visualization
- WebGL rendering

**5.5: GPU Acceleration**
- Deferred until CPU performance is sufficient
- Consider Burn framework (autodiff + fusion)
- See `reference/burn-evaluation.md` for analysis

---

## Maintenance Plan (Current Sprint)

### Phase A: Quick Wins ⬜

**Goal**: Reorganize documentation for better LLM token efficiency.

**Tasks**:
1. Create `plans/` folder
2. Move 14 planning documents from `reference/` to `plans/`
   - 5 GPU plans (gpu-*.md)
   - 5 Performance plans (perf-*.md)
   - 3 Serialization plans (serialisation-*.md)
   - 1 Gradient clipping (gradient_clipping.md)
3. Create `plans/INDEX.md` with categorized listing
4. Update `AGENTS.md` with token-saving guidelines
5. Keep in `reference/`: notebook, Python script, visualizations

**Exit criteria**:
- ✅ All planning docs in `plans/` folder
- ✅ `plans/INDEX.md` exists
- ✅ `AGENTS.md` updated
- ✅ `reference/` contains only reference materials

**Estimated effort**: 30-60 minutes

---

### Phase B: Complete Phase 2.1a ⬜

See Phase 2.1a section above.

---

### Phase C: Deep Refactoring ⬜

**Goal**: Clean codebase - remove dead code, rename files.

**Tasks**:

1. **Audit legacy modules** (3377 lines)
   - `phase_0_1.rs` - KEEP (still used) → rename to `gates.rs`
   - `phase_0_2.rs` - likely dead, check and remove
   - `phase_0_3.rs` - likely dead, check and remove
   - `phase_1_1.rs` - superseded by grid.rs/perception.rs, remove
   - `trainer.rs` - superseded by training.rs, remove

2. **Clean binaries** (17 files)
   - Keep: `train_checkerboard.rs`, `train_gol.rs`
   - Archive or delete: 8 `debug_*.rs`, 3 `test_*.rs`
   - Add: `visualize_checkerboard.rs`, `test_generalization.rs`

3. **Update imports**
   - Fix all imports after renaming
   - Update lib.rs

4. **Update documentation**
   - `agents/INDEX.md` - reflect new file structure
   - `agents/implementation-log.md` - compact to <150 lines
   - Remove outdated information

**Exit criteria**:
- ✅ No `phase_*.rs` files (except renamed `gates.rs`)
- ✅ <10 binaries in `src/bin/`
- ✅ All 148+ tests pass
- ✅ `agents/INDEX.md` accurate
- ✅ Implementation log compacted

**Estimated effort**: 4-6 hours

---

## Development Workflow

1. **Read requirements**: Check phase description in this plan
2. **Create branch**: Use descriptive names (e.g., `feature/phase-2.2-async`)
3. **Create implementation plan**: Detailed step-by-step in `plans/phase-*.md`
4. **Write tests first**: Unit tests before implementation
5. **Implement**: Follow test-driven development
6. **Verify exit criteria**: All must be met
7. **Document**: Update implementation-log.md (keep it compact)
8. **Commit & PR**: Never commit to main directly
9. **Review & merge**: Human approval required

---

## Key Technical Details

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 0.05 | AdamW |
| β1 | 0.9 | AdamW momentum |
| β2 | 0.99 | **Critical**: Not 0.999 - escapes local minima |
| Gradient clip | 100.0 | Prevents explosion |
| Fire rate | 0.6 | Async training only |
| Pass-through init | logit=10.0 | Gate index 3 (operation A) |

### Connection Types

- **first_kernel**: Center cell vs 8 neighbors (perception layer 1)
- **unique**: Unique pair connections (all other layers)

### Model Architectures

| Model | Perception | Update | Total Gates |
|-------|------------|--------|-------------|
| GoL full | 16×[9→8→4→2→1]=240 | [17→128×10→1]=1407 | 1647 |
| Checkerboard full | 16×[9→8→4→2]=224 | [264→256×10→8]=2816 | 3040 |
| Checkerboard small | 16×[9→8→4→2]=224 | [264→256×4→8]=1272 | 1496 |

### Soft vs Hard Mode

- **Soft**: `softmax(logits)` weighted average of all 16 operations → differentiable
- **Hard**: `argmax(logits)` selects single operation → discrete, fast inference

### Gradient Scaling

- **Critical**: Use `scale = 1.0` (raw sum loss, no averaging)
- Matches reference implementation exactly
- Averaging breaks training dynamics

---

## Critical Success Factors

1. ✅ **N-bit from start**: No refactoring for multi-channel
2. ✅ **Verification first**: Never proceed with failing tests
3. ✅ **Match reference**: Compare outputs layer-by-layer when debugging
4. ✅ **GoL is validation**: Real value is multi-channel experiments (Phases 2.x)
5. ✅ **Document learnings**: Capture insights in implementation-log.md
6. ✅ **Incremental testing**: C=1 → C=8 → C=64 → C=128

---

## Risk Mitigation

### Before starting any phase:
- Create git branch
- Ensure all tests pass on main
- Commit any unsaved training results

### During implementation:
- Run tests frequently: `cargo test --lib`
- Keep commits small and atomic
- Can always revert if needed

### Before marking phase complete:
- All exit criteria met
- All tests pass
- Documentation updated
- Results reproducible

---

## Long-Running Tasks Protocol

For training runs that take >30 minutes:

1. **Don't run yourself** - provide command to human
2. **Complete other work first** - maximize independence
3. **Clear instructions** - explain what to look for
4. **Wait for feedback** - adjust based on results

Example:
```bash
# Training will take ~30 minutes
cargo run --bin train_checkerboard --release

# Expected: 100% accuracy around epoch 500
# Report final accuracy and check checkerboard_sync_log.csv
```

---

## Test Commands

```bash
# Unit tests
cargo test --lib                    # All tests
cargo test --lib -- --nocapture     # With output
cargo test grid::tests              # Specific module

# Release builds
cargo build --release               # Always for training
cargo run --bin train_checkerboard --release

# Training with options
cargo run --bin train_checkerboard --release -- --small
cargo run --bin train_checkerboard --release -- --epochs=500
cargo run --bin train_checkerboard --release -- --log-interval=10
```

---

**Last Updated**: 2026-01-09

**Current Test Count**: 148 passing
**Current Focus**: Phase 2.1a (analysis & documentation)
**Next Phase**: Phase 2.2 (async checkerboard)
