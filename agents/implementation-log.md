# Logicars Implementation Log

> Compact status log for LLM agents. Full development history archived in git commits.

**Last Updated**: 2026-01-09
**Current Focus**: Phase A - Quick Maintenance Wins
**Next Phase**: Phase 2.1a - Checkerboard Analysis & Documentation

---

## 🎯 Current State

### Major Milestone: Phase 2.1 Complete ✅

**Checkerboard training succeeded!** (2026-01-09)
- Epoch 512: First 100% hard accuracy
- Epochs 512-637: Maintained 100% accuracy, hard_loss=0.0
- Total training time: ~35 minutes (2121 seconds)
- Model: 3040 gates (16 kernels, 16-layer update network)
- Results in: `checkerboard_sync_log.csv`

This validates the N-bit architecture generalizes beyond GoL (C=1) to multi-channel tasks (C=8).

### What's Done

| Phase | Description | Status | Date |
|-------|-------------|--------|------|
| 0.x | Gate primitives, backprop | ✅ Complete | 2026-01-05 |
| 1.x | N-bit architecture, GoL 99.41% | ✅ Complete | 2026-01-06 |
| 2.1 | Checkerboard C=8 training | ✅ Complete | 2026-01-09 |
| Perf 1,2,3,5 | Release profile, rayon parallel, caching | ✅ Complete | 2026-01-08 |

**Test Count**: 148 passing
**Architecture**: Stable, supports C=1 to C=128 channels

---

## 📋 Current Task: Phase A (Quick Maintenance Wins)

**Goal**: Reorganize documentation for better LLM token efficiency.

**See**: `agents/plan.md` → "Maintenance Plan (Current Sprint)" → "Phase A"

### Tasks (30-60 minutes total)

1. **Create `plans/` folder**
2. **Move 14 planning documents** from `reference/` to `plans/`:
   - 5 GPU plans: `gpu-*.md`
   - 5 Performance plans: `perf-*.md`
   - 3 Serialization plans: `serialisation-*.md`
   - 1 Other: `gradient_clipping.md`
3. **Create `plans/INDEX.md`** with categorized listing
4. **Update `AGENTS.md`** with token-saving guidelines
5. **Keep in `reference/`**: notebook, Python script, visualizations, burn-evaluation.md

### Exit Criteria

- ✅ All planning docs in `plans/` folder
- ✅ `plans/INDEX.md` exists and is clear
- ✅ `AGENTS.md` has token-saving section
- ✅ `reference/` contains only reference materials (not planning docs)

### What to Pay Attention To

- **Don't move** the reference implementation files (`.ipynb`, `.py`)
- **Don't move** visualization files (`.png`, `.svg`)
- **Don't move** `reference/README.md` or `burn-evaluation.md`
- **Do move** all `*-plan.md` and `*-phase*.md` files
- Create a clear index so future LLMs know where to find things

---

## 📖 After Phase A: What's Next

### Phase B: Complete Phase 2.1a Analysis (8-10 hours)

**Goal**: Scientific analysis of trained checkerboard model

**See**: `agents/plan.md` → "Active Phase" → "Phase 2.1a"

**Tasks**:
1. Gate distribution analysis (which gates were learned?)
2. Visual reconstruction GIF (20-step rollout animation)
3. Generalization test (64×64 grid, >95% accuracy)
4. Documentation update

### Phase C: Deep Refactoring (4-6 hours)

**Goal**: Clean codebase - remove dead code, rename legacy files

**See**: `agents/plan.md` → "Maintenance Plan" → "Phase C"

**Tasks**:
1. Audit and remove legacy `phase_*.rs` files (keep phase_0_1.rs, rename to gates.rs)
2. Clean up 17 binaries (keep core training, remove debug_*)
3. Update `agents/INDEX.md` to reflect new structure

---

## 🔧 Technical Quick Reference

### Hyperparameters (Critical Values)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 0.05 | AdamW |
| β2 | **0.99** | **CRITICAL**: Not 0.999 - escapes local minima |
| Gradient clip | 100.0 | Prevents explosion |
| Gradient scale | **1.0** | **CRITICAL**: Raw sum, no averaging |
| Pass-through init | 10.0 | At gate index 3 (operation A) |
| Batch size (CB) | 2 | Checkerboard sync |
| Loss channel (CB) | 0 | **CRITICAL**: Channel 0 only, not all channels |

### Model Architectures

| Model | Perception | Update | Total Gates |
|-------|------------|--------|-------------|
| GoL full | 16×[9→8→4→2→1]=240 | [17→128×10→1]=1407 | 1647 |
| Checkerboard full | 16×[9→8→4→2]=224 | [264→256×10→8]=2816 | 3040 |
| Checkerboard small | 16×[9→8→4→2]=224 | [264→256×4→8]=1272 | 1496 |

### Key Code Locations

| Component | File | Key Functions |
|-----------|------|---------------|
| Grid & neighborhoods | `src/grid.rs` | `NGrid`, `neighborhood()` |
| Perception | `src/perception.rs` | `PerceptionModule::forward_soft()` |
| Update network | `src/update.rs` | `UpdateModule`, `DiffLogicCA` |
| Training loop | `src/training.rs` | `TrainingLoop::train_step_batch()` |
| Checkerboard task | `src/checkerboard.rs` | `create_checkerboard()`, loss functions |
| Serialization | `src/circuit.rs` | `HardCircuit::from_soft()`, `save()/load()` |
| Gate primitives | `src/phase_0_1.rs` | `BinaryOp`, `ProbabilisticGate` |

**Navigation**: Always check `agents/INDEX.md` first for file:line references.

---

## 🎓 Key Learnings (Don't Repeat These Mistakes)

1. **β2=0.99** is critical - 0.999 gets stuck in local minima
2. **Loss channel matters** - For checkerboard, only channel 0 has the pattern. Channels 1-7 are working memory.
3. **Gradient scale=1.0** - Reference uses raw sum, not averaging. Averaging breaks training.
4. **Perception output ordering** - Must be (c, s, k) not (c, k, s). See `perception.rs:373`
5. **Batch training helps** - batch_size=2 provides gradient variance, shortens convergence plateau
6. **200-epoch plateau is normal** - Gates escaping pass-through initialization (logit=10.0)

---

## 🚀 Common Commands

```bash
# Tests
cargo test --lib                    # All 148 tests
cargo test --lib -- --nocapture     # With output

# Training
cargo run --bin train_checkerboard --release
cargo run --bin train_checkerboard --release -- --epochs=500
cargo run --bin train_gol --release -- --full

# Build
cargo build --release               # Always use release for training
```

---

## 📚 Documentation Structure

**For high-level planning**:
- `agents/plan.md` - Full roadmap, phase descriptions, exit criteria
- `agents/INDEX.md` - Code navigation (file:line references)
- `AGENTS.md` - Getting started, workflow

**For detailed implementation** (to be created):
- `plans/phase-2.1a-analysis.md` - Gate analysis, GIF export, generalization
- `plans/phase-2.5-pyo3.md` - PyO3 bindings specification
- `plans/perf-*.md` - Performance optimization details (being moved from reference/)

**Reference materials**:
- `reference/diffLogic_CA.ipynb` - Ground truth (ALWAYS TRUST THIS)
- `reference/difflogic_ca.py` - Compacted Python (may have errors)
- `reference/README.md` - Paper summary
- `reference/burn-evaluation.md` - Future GPU strategy

---

## 🔮 Roadmap Ahead

**Immediate** (this week):
- Phase A: Documentation reorganization (30-60 min)
- Phase B: Checkerboard analysis (8-10 hours)
- Phase C: Code refactoring (4-6 hours)

**Next experiments**:
- Phase 2.2: Async checkerboard (self-healing)
- Phase 2.3: Growing Lizard (C=128, morphogenesis)

**Strategic pivot**:
- Phase 2.5: Minimal PyO3 bindings (1-2 weeks) ← **Before Colored G**
- Phase 2.4: Colored G (C=64, 8-color palette) ← **After PyO3**

**Rationale**: Colored G needs proper visualization (matplotlib). Building PyO3 after Lizard provides better development experience and informed API design.

**Long-term**:
- Phase 5.1: Production PyO3 API
- Phase 5.2: extendr R bindings (both Python AND R - yes, it's possible!)
- Phase 5.3: Visualization tools
- Phase 5.4: WASM demo

---

## ⚠️ Critical Notes for Next Agent

1. **Read `agents/plan.md` FIRST** - It has the current phase details and exit criteria
2. **Current task is Phase A** - Just documentation reorganization, no code changes
3. **Training already succeeded** - Don't re-run training unless specifically needed for Phase 2.1a
4. **Use `agents/INDEX.md`** - File:line references save tokens vs grepping
5. **Perf optimizations are DONE** - Phases 1,2,3,5 complete (rayon parallelization, caching)
6. **Never commit to main** - Always create branch, open PR
7. **Keep this log compact** - Full history is in git, this is just current state

---

**Status**: ✅ Phase 2.1 complete, ready for Phase A maintenance
**Branch**: Currently on `docs/plan-rewrite` (plan.md rewrite PR open)
**Next**: Start Phase A tasks after plan.md PR is merged
