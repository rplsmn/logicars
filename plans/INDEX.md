# Plans Directory Index

This directory contains detailed implementation plans, optimization strategies, and architectural decisions for the Logicars project.

**For quick navigation**: Use this index to find the right plan document before diving into details.

---

## Performance Optimization Plans

Implementation plans for CPU performance improvements (Phases 1-5 complete, ~10-15x speedup achieved).

| File | Status | Description |
|------|--------|-------------|
| `perf-phase1-release-profile.md` | ✅ Complete | Release profile tuning (LTO, codegen-units) |
| `perf-phase2-parallel-backward.md` | ✅ Complete | Rayon parallelization of backward pass |
| `perf-phase3-batch-parallel.md` | ✅ Complete | Batch parallelization with rayon |
| `perf-phase4-f32-precision.md` | ⬜ Documented | f32 vs f64 precision trade-offs (not implemented) |
| `perf-phase5-softmax-caching.md` | ✅ Complete | Softmax probability caching with RwLock |
| `performance-improvement-profiling-analysis.md` | 📊 Analysis | Profiling data and bottleneck analysis |
| `performance-improvements-options.md` | 📊 Analysis | Performance optimization options overview |

**Quick reference**: Phases 1, 2, 3, 5 are complete. Phase 4 (f32) is documented but not implemented.

---

## GPU Acceleration Plans

Plans for GPU implementation using WGPU (PAUSED - CPU implementation is active track).

| File | Status | Description |
|------|--------|-------------|
| `gpu-plan.md` | 🚧 Overview | High-level GPU acceleration strategy |
| `gpu-phase1-plan.md` | ✅ Merged | Basic WGPU setup (code in `src/gpu/`) |
| `gpu-phase2-plan.md` | ⬜ Planned | Forward pass on GPU |
| `gpu-phase3-plan.md` | ⬜ Planned | Backward pass on GPU |
| `gpu-phase4-plan.md` | ⬜ Planned | Full training loop on GPU |

**Status**: Phase 1 merged but frozen. See `reference/burn-evaluation.md` for future GPU strategy (considering Burn framework).

---

## Serialization Plans

Model serialization and export (Phase 3.2 complete - JSON save/load works).

| File | Status | Description |
|------|--------|-------------|
| `serialisation-plan.md` | 📋 Overview | Serialization strategy overview |
| `serialisation-phase1-plan.md` | ✅ Complete | JSON export with serde |
| `serialisation-phase2-plan.md` | ✅ Complete | HardCircuit save/load |

**Implementation**: See `src/circuit.rs` for working JSON serialization.

---

## Phase Implementation Plans

Detailed task breakdowns for specific phases.

| File | Status | Description |
|------|--------|-------------|
| `phase-2.1a-analysis.md` | 🚧 Active | Checkerboard analysis, GIF generation, generalization testing |

---

## Other Technical Documents

| File | Description |
|------|-------------|
| `gradient_clipping.md` | Gradient clipping implementation notes (value: 100.0) |

---

## When to Read These Plans

### Starting performance work?
- Read `performance-improvements-options.md` for overview
- Check which phases are complete vs planned
- Don't re-implement completed optimizations

### GPU work?
- **Don't** - GPU work is frozen
- Read `reference/burn-evaluation.md` for future strategy
- Focus on CPU implementation

### Need to export models?
- Serialization is done - see `src/circuit.rs`
- Read `serialisation-phase1-plan.md` for format details

### Debugging training?
- Check `gradient_clipping.md` for hyperparameters
- Performance plans have profiling data

---

## Navigation Tips for LLMs

1. **Check this index first** before reading full plan documents
2. **Use agents/INDEX.md** for code locations (file:line references)
3. **Completed phases**: Don't re-read unless debugging specific feature
4. **Active work**: Focus on agents/plan.md and agents/implementation-log.md
5. **Token efficiency**: Only read plans/ files when working on that specific subsystem

---

**See also**:
- `agents/plan.md` - Current phase and roadmap
- `agents/implementation-log.md` - Recent progress and next steps
- `agents/INDEX.md` - Code navigation (file:line references)
- `reference/` - Reference implementation and visualizations
