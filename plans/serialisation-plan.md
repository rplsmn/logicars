# Model Serialisation Plan for Logicars

## Summary

This document outlines the plan to add full model serialisation to Logicars, enabling:
1. **Checkpoint saving**: Save training state to resume later
2. **Model export**: Export trained soft models for sharing
3. **Resume training**: Load checkpoint and continue training
4. **Inference-only export**: Already implemented via `HardCircuit`

---

## Current State

### What Exists

| Component | Serialisation | Notes |
|-----------|---------------|-------|
| `HardCircuit` | ✅ JSON | Discrete gates only, for inference |
| `DiffLogicCA` (soft) | ❌ None | Full model with learnable logits |
| Optimizer state | ❌ None | AdamW momentum buffers |
| Training progress | ❌ None | Epoch, best accuracy, RNG state |

### Limitation

The `demo_inference` branch demonstrates the problem: every run must retrain from scratch because there's no way to save/load the soft model. With training runs taking 8+ hours, this is a critical blocker.

---

## Requirements

### Must Have (Phase 1)
1. Save/load `DiffLogicCA` model weights (gate logits)
2. Save/load optimizer state (AdamW m, v buffers)
3. Save/load training metadata (epoch, best_accuracy, rng_state)
4. File format: JSON (human-readable) or bincode (compact)

### Should Have (Phase 2)
1. Version compatibility checking
2. Compression for large models
3. Model metadata (creation date, config, accuracy)

### Nice to Have (Phase 3)
1. ONNX export for interoperability
2. Streaming save/load for memory efficiency
3. Encrypted checkpoints

---

## Design Decisions

### Format Choice: JSON + bincode hybrid

| Format | Pros | Cons | Use Case |
|--------|------|------|----------|
| JSON | Human-readable, debuggable | Large, slow | Configs, metadata |
| bincode | Fast, compact, Rust-native | Not readable | Weight arrays |
| MessagePack | Cross-language, compact | Extra dependency | Not needed |

**Decision**: Use `bincode` for weights, JSON for metadata. This gives fast I/O for large weight arrays while keeping configs readable.

### File Structure

```
checkpoint.logicars/
├── metadata.json        # Version, epoch, config, accuracy
├── model.bin           # Gate logits (bincode)
├── optimizer.bin       # AdamW state (bincode)
└── rng_state.bin       # RNG for reproducibility
```

Single-file alternative: `.logicars` file with embedded tar or zip.

**Decision**: Start with directory-based (simpler), add single-file in Phase 2.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Checkpoint                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Metadata    │  │ Model       │  │ Optimizer           │ │
│  │ (JSON)      │  │ Weights     │  │ State               │ │
│  │             │  │ (bincode)   │  │ (bincode)           │ │
│  │ - version   │  │             │  │                     │ │
│  │ - epoch     │  │ - perception│  │ - m (momentum)      │ │
│  │ - config    │  │   logits[]  │  │ - v (velocity)      │ │
│  │ - accuracy  │  │ - update    │  │ - t (timestep)      │ │
│  │ - timestamp │  │   logits[]  │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     TrainingLoop                             │
│                                                              │
│  TrainingLoop::save_checkpoint(path)                        │
│  TrainingLoop::load_checkpoint(path) -> Result<TrainingLoop>│
│                                                              │
│  DiffLogicCA::save(path)  / DiffLogicCA::load(path)         │
│  AdamW::save(path)        / AdamW::load(path)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase Overview

| Phase | Description | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 1.1 | Model weight serialisation | 2-3h | serde, bincode |
| 1.2 | Optimizer state serialisation | 1-2h | Phase 1.1 |
| 1.3 | Checkpoint integration | 2-3h | Phase 1.2 |
| 1.4 | CLI integration | 1-2h | Phase 1.3 |
| 2.1 | Version compatibility | 1-2h | Phase 1.x |
| 2.2 | Single-file checkpoints | 2-3h | Phase 1.x |
| 3.1 | ONNX export (stretch) | 4-8h | Phase 1.x |

**Total estimated effort**: 8-12 hours for Phase 1, 3-5 hours for Phase 2

---

## Implementation Order

### Phase 1: Core Serialisation (PRIORITY)

See [`serialisation-phase1-plan.md`](serialisation-phase1-plan.md) for detailed implementation.

**Goal**: Save and resume training runs.

1. **1.1 Model Weights**: Derive Serialize/Deserialize for gates, layers, modules
2. **1.2 Optimizer State**: Save/load AdamW m, v buffers  
3. **1.3 Checkpoint System**: Unified save/load for training state
4. **1.4 CLI Integration**: Add `--save-checkpoint` and `--load-checkpoint` options

### Phase 2: Robustness

See [`serialisation-phase2-plan.md`](serialisation-phase2-plan.md) for detailed implementation.

**Goal**: Production-ready checkpointing.

1. **2.1 Versioning**: Add format version, check compatibility on load
2. **2.2 Single File**: Package checkpoint as single `.logicars` file

### Phase 3: Interoperability (Stretch)

**Goal**: Export for use outside Rust.

1. **3.1 ONNX**: Export hard circuit to ONNX format

---

## Dependencies to Add

```toml
[dependencies]
bincode = "1.3"  # Already have serde, serde_json
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Format changes break old checkpoints | High | Medium | Version field + migration code |
| Large checkpoints slow to save | Medium | Low | bincode is fast, async save in bg |
| Optimizer state mismatch | Medium | High | Validate shapes on load |

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Can save a training run at epoch N
- [ ] Can load and resume from epoch N
- [ ] Training continues correctly (loss matches expected trajectory)
- [ ] Tests verify save/load round-trip

### Phase 2 Complete When:
- [ ] Old checkpoints load with warning if version differs
- [ ] Single-file checkpoints work
- [ ] Compression reduces file size by >50%

---

## Next Steps

1. Read `serialisation-phase1-plan.md` for Phase 1 implementation details
2. Create feature branch `feature/model-serialisation`
3. Implement Phase 1.1 (model weights) first
4. Test with short training run (save at epoch 50, load, continue to 100)
