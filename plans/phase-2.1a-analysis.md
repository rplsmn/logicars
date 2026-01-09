# Phase 2.1a: Checkerboard Analysis & Documentation

**Goal**: Complete scientific analysis of trained checkerboard model before moving to Phase 2.2 (async).

**Prerequisites**:
- Phase 2.1 complete: 100% accuracy achieved (epoch 512+)
- Training log: `checkerboard_sync_log.csv`
- HardCircuit infrastructure in `circuit.rs` (save/load, gate_distribution)

---

## Tasks

### Task 1: Add Model Saving to Training Binary

**File**: `src/bin/train_checkerboard.rs`

**What to do**:
1. Add `--save=PATH` CLI option for saving trained model
2. After training loop ends, convert model to `HardCircuit` and save
3. Also save on best accuracy improvement (checkpoint)

**Implementation**:
```rust
// At end of training loop
use logicars::HardCircuit;

let circuit = HardCircuit::from_soft(&training_loop.model);
circuit.save("checkerboard_model.json")?;
println!("Saved model to checkerboard_model.json");
```

**Test**: Run training for 1 epoch with `--save=test.json`, verify file exists and is valid JSON.

**Exit criteria**:
- ✅ `--save=PATH` option works
- ✅ Model saved as JSON after training
- ✅ JSON can be loaded back with `HardCircuit::load()`

---

### Task 2: Create Analysis Binary

**File**: `src/bin/analyze_checkerboard.rs`

**What to do**:
1. Load trained model from JSON
2. Print gate distribution (count of each of 16 operations)
3. Calculate active vs pass-through ratio
4. Export analysis to CSV

**Implementation outline**:
```rust
use logicars::{HardCircuit, BinaryOp};

fn main() {
    let circuit = HardCircuit::load("checkerboard_model.json").unwrap();
    
    // Gate distribution
    let dist = circuit.gate_distribution();
    println!("Gate Distribution:");
    for (i, count) in dist.iter().enumerate() {
        let op = BinaryOp::ALL[i];
        println!("  {:?}: {}", op, count);
    }
    
    // Active vs pass-through
    let total = circuit.total_gate_count();
    let active = circuit.active_gate_count();
    println!("\nActive gates: {} / {} ({:.1}%)", 
             active, total, 100.0 * active as f64 / total as f64);
}
```

**Test**: Load a model, verify output matches manual inspection.

**Exit criteria**:
- ✅ Binary loads model and prints gate distribution
- ✅ Active/pass-through ratio displayed
- ✅ Results exportable to CSV

---

### Task 3: Generate Rollout GIF

**File**: `src/bin/visualize_checkerboard.rs`

**What to do**:
1. Load trained model
2. Generate random seed
3. Run 20 steps, capturing each frame
4. Export as animated GIF (use `image` crate)

**Implementation notes**:
- Add `image` and `gif` crates to Cargo.toml (dev-dependencies or optional)
- Each frame: convert channel 0 to grayscale image
- Scale up pixels (e.g., 8x) for visibility
- Frame delay: 100-200ms

**Output**: `checkerboard_rollout.gif`

**Test**: Verify GIF shows pattern emergence from random to checkerboard.

**Exit criteria**:
- ✅ GIF generated with 20+ frames
- ✅ Pattern visibly emerges from noise
- ✅ Final frame matches expected checkerboard

---

### Task 4: Generalization Testing

**File**: `src/bin/test_generalization.rs` (or extend `analyze_checkerboard.rs`)

**What to do**:
1. Load 16×16 trained model
2. Test on 32×32, 64×64, 128×128 grids
3. Report accuracy at each size
4. Test with different step counts (20, 40, 60 steps)

**Expected results**:
- 64×64: >95% accuracy (exit criterion)
- Larger grids may need more steps

**Test**: Verify accuracy calculation is correct.

**Exit criteria**:
- ✅ 64×64 grid: >95% accuracy
- ✅ Results documented with step counts

---

### Task 5: Documentation Update

**Files**:
- `agents/implementation-log.md`
- `agents/plan.md`

**What to do**:
1. Record gate distribution findings
2. Document generalization results
3. Add GIF to reference/ or link in docs
4. Mark Phase 2.1 as fully complete in plan.md
5. Update implementation-log.md (keep <100 lines)

**Exit criteria**:
- ✅ All results documented
- ✅ Phase 2.1 marked ✅ in plan.md
- ✅ implementation-log.md updated and compact

---

## Implementation Order

1. **Task 1** (model saving) - enables all other tasks
2. **Task 2** (analysis) - quick win, useful data
3. **Task 4** (generalization) - validates model quality
4. **Task 3** (GIF) - visual proof, may need image crate setup
5. **Task 5** (docs) - after all results collected

---

## Dependencies

**Rust crates needed**:
```toml
[dependencies]
image = { version = "0.25", optional = true }

[features]
visualize = ["image"]
```

**Note**: Keep visualization dependencies optional to avoid bloating the core library.

---

## Commands

```bash
# Run with model saving (requires training first)
cargo run --bin train_checkerboard --release -- --epochs=1 --save=test_model.json

# Analysis (after training with save)
cargo run --bin analyze_checkerboard --release

# GIF generation
cargo run --bin visualize_checkerboard --release --features visualize

# Generalization test
cargo run --bin test_generalization --release
```

---

## Success Criteria (Phase 2.1a Complete)

- ✅ Gate distribution analysis saved (CSV or printed)
- ✅ Animated GIF generated showing rollout
- ✅ 64×64 generalization validated (>95% accuracy)
- ✅ All results documented in implementation-log.md
- ✅ Phase 2.1 marked ✅ COMPLETE in plan.md

---

## Notes

- Training already succeeded (100% accuracy at epoch 512)
- Do NOT re-run full training - use saved model or run minimal epochs
- GIF generation is throwaway code - doesn't need to be perfect
- Focus on validating the model works as expected before async phase
