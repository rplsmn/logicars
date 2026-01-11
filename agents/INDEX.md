# Code Index (for LLM navigation)

**SEARCH THIS FILE FIRST** before grepping the codebase. Use `file:line` references to jump directly.

## Quick Reference

### Key Constants
```
training.rs:21    FIRE_RATE = 0.6 (async cell update probability)
training.rs:24    GRADIENT_CLIP = 100.0
checkerboard.rs:23-28  CHECKERBOARD_* (8 channels, 16 kernels, 16×16 grid, 2×2 squares, 20/50 steps)
```

### Architecture Flow
```
Input Grid → PerceptionModule → UpdateModule → Output Grid
     ↓              ↓                ↓
   NGrid      264 features      8 channels
   16×16     (8 center +       back to grid
   8 chan     16k×2×8 kernel)
```

---

## Module Overview (by purpose)

| File | Purpose | Key Types |
|------|---------|-----------|
| `grid.rs` | Grid storage, neighborhoods, boundaries | `NGrid`, `NNeighborhood`, `BoundaryCondition` |
| `perception.rs` | Perception kernels, feature extraction | `PerceptionModule`, `PerceptionKernel`, `GateLayer` |
| `update.rs` | Update network, full CA model | `UpdateModule`, `DiffLogicCA`, `DiffLogicCATrainer` |
| `training.rs` | BPTT training loop, optimizer | `TrainingLoop`, `TrainingConfig`, `SimpleRng` |
| `checkerboard.rs` | Checkerboard task setup | `create_checkerboard_model()`, loss/accuracy fns |
| `gates.rs` | Single gate operations | `BinaryOp`, `ProbabilisticGate` |
| `circuit.rs` | Hard (discrete) inference | `HardCircuit`, `HardPerception`, `HardUpdate` |
| `optimizer.rs` | AdamW optimizer | `AdamW` |

---

## Function Index

### Grid Operations (`grid.rs`)
```
40   NGrid::new(w, h, c, boundary)      Create grid
53   NGrid::periodic(w, h, c)           Periodic boundary grid
58   NGrid::non_periodic(w, h, c)       Zero-padded boundary grid
110  NGrid::get(x, y, ch) -> f64        Get cell value (zero-pads OOB for non-periodic)
118  NGrid::set(x, y, ch, val)          Set cell value
169  NGrid::neighborhood(x, y)          Extract 3×3 neighborhood
265  NNeighborhood - 9 cells × C channels, reading order [NW,N,NE,W,C,E,SW,S,SE]
```

### Perception (`perception.rs`)
```
312  PerceptionModule::new(channels, kernels, layer_sizes)
373  PerceptionModule::forward_soft()    Soft forward (training)
462  PerceptionModule::backward()        Backward pass
190  PerceptionKernel - processes one 3×3 neighborhood offset
138  GateLayer - single layer of probabilistic gates
```

### Update Network (`update.rs`)
```
35   UpdateModule::new(input_dim, layer_sizes)
98   UpdateModule::forward_soft()        Soft forward
160  UpdateModule::backward()            Backward pass
274  DiffLogicCA - combines Perception + Update
281  DiffLogicCA::forward_soft/hard()    Full CA step
```

### Training (`training.rs`)
```
155  TrainingLoop::new(model, config)
196  TrainingLoop::train_step()          One training iteration
225  TrainingLoop::run_steps()           Multi-step rollout (inference)
495  TrainingLoop::backward_through_time()  BPTT implementation
60   TrainingConfig::gol()               GoL preset
75   TrainingConfig::checkerboard_sync() Checkerboard preset
```

### Checkerboard Task (`checkerboard.rs`)
```
41   create_checkerboard(size, sq, ch)   Create target pattern
67   create_random_seed(size, ch, rng)   Random initial state
144  create_checkerboard_model()         Full model (3040 gates)
151  create_small_checkerboard_model()   Small model (1496 gates)
165  compute_checkerboard_loss()         MSE on channel 0
187  compute_checkerboard_accuracy()     Hard accuracy on channel 0
```

### Gate Operations (`gates.rs`)
```
12   BinaryOp enum - all 16 binary operations
70   ALL: [BinaryOp; 16] - ordered by truth table value
91   ProbabilisticGate - learnable gate with 16 logits
115  ProbabilisticGate::forward_soft()   Soft output (training)
134  ProbabilisticGate::forward_hard()   Hard output (inference)
150  ProbabilisticGate::backward()       Compute gradients
```

---

## Key Implementation Details

### Soft vs Hard Mode
- **Soft**: `softmax(logits)` weighted average of all 16 ops → differentiable
- **Hard**: `argmax(logits)` selects single op → discrete/fast

### Pass-Through Gate Init
- `gates.rs:102`: logits[12] = 10.0 (op A at index 12 in our ordering)

### Perception Output Layout
- `perception.rs:373`: Order is (c, s, k) = channel, sobel, kernel
- Output size: `channels + kernels × 2 × channels` = 8 + 16×2×8 = 264

### Boundary Handling  
- `grid.rs:110-120`: NonPeriodic returns 0.0 for out-of-bounds (zero-padding)
- `grid.rs:152-155`: Periodic uses `rem_euclid` for wrapping

### Gradient Scaling
- `training.rs:573-580`: Uses `scale = 1.0` (raw sum, no averaging) to match reference

### Async Training (Fire Rate Masking)
- `training.rs:441-511`: `forward_grid_soft_async()` - forward pass with fire rate masking
- `training.rs:641-722`: `compute_sample_gradients_async()` - async gradient computation
- `training.rs:855-927`: Backward pass handles fire_mask (skips unfired cells)

---

## Binaries

| Binary | Purpose |
|--------|---------|
| `train_checkerboard` | Phase 2.1 sync checkerboard training with --save, --log options |
| `train_checkerboard_async` | Phase 2.2 async training with fire rate masking, self-healing test |
| `analyze_checkerboard` | Phase 2.1a gate distribution (CSV output) |
| `visualize_checkerboard` | Phase 2.1a GIF generation (--size, --steps options) |
| `train_gol` | Game of Life training |
| `test_generalization` | Test model on larger grids |

---

## Test Commands
```bash
cargo test --lib                    # All unit tests (126 tests)
cargo test --lib -- --nocapture     # With output
cargo test grid::tests              # Specific module
cargo test -- async                 # Async training tests only
cargo build --release               # Release build
```
