# Reference Implementation

This folder contains a snapshot of the reference Python/JAX implementation for comparison and guidance.

## Source

- **Paper**: [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/)
- **Original Notebook**: [Google Colab](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/diffLogic_CA.ipynb)
- **Repository**: https://github.com/google-research/self-organising-systems

## License

Copyright 2025 Google LLC - Licensed under Apache License 2.0

## Contents

- `difflogic_ca.py` - Extracted Python code with ALL experiment hyperparameters
- `diffLogic_CA.ipynb` - Original Colab notebook (full reference)

### What's in `difflogic_ca.py`

**Core Logic** (all gates/operations):
- `bin_op_all_combinations`, `bin_op_s`, `decode_soft/hard`

**Network Construction**:
- `get_moore_connections`, `get_unique_connections`
- `init_gates`, `init_gate_layer`
- `init_logic_gate_network`, `init_perceive_network`, `init_diff_logic_ca`

**Execution**:
- `run_layer`, `run_update`, `run_perceive`, `run_circuit`
- `get_grid_patches`, `run_async`, `run_sync`, `run_iter_nca`

**Training**:
- `loss_f`, `init_state`, `train_step`

**Data Generation**:
- `gol_step` - Single GoL step
- `simulate_gol_batch` - Multi-step GoL simulation
- `generate_all_3x3_neighborhoods` - All 512 binary 3x3 configs
- `sample_gol_batch` - Sample training batches from trajectories
- `create_checkerboard` - Generate checkerboard target pattern

**Experiment Hyperparameters**:
- `GOL_HYPERPARAMS`, `CHECKERBOARD_SYNC_HYPERPARAMS`, `CHECKERBOARD_ASYNC_HYPERPARAMS`
- `GROWING_LIZARD_HYPERPARAMS`, `COLORED_G_HYPERPARAMS`

---

## Paper Experiments Overview

The paper presents 5 experiments of increasing complexity. **All experiments use the same core architecture** (perception + update modules), differing only in:
- State size (channels/bits per cell)
- Number of perception kernels
- Network depth
- Training steps

| Experiment | Channels | Kernels | Grid | Steps | Key Property |
|------------|----------|---------|------|-------|--------------|
| Game of Life | 1 | 16 | 128x128 | 1 | Binary CA, all 512 configs |
| Checkerboard (Sync) | 8 | 16 | 16x16 | 20 | Pattern generation |
| Checkerboard (Async) | 8 | 16 | 16x16 | 50 | Fault tolerance |
| Growing Lizard | 128 | 4 | 20x20 | 12 | Complex shape |
| Colored G | 64 | 4 | 20x20 | 15 | RGB, 927 gates |

---

## Experiment 1: Game of Life

**Goal**: Learn Conway's Game of Life rule from examples

### Architecture
- **Perception**: 16 parallel kernels, each `[9→8→4→2→1]`
- **Update**: 18 layers `[17→128×10→64→32→16→8→4→2→1]`
- **Total active gates**: ~336

### Training
- Learning rate: 0.05
- Batch size: 20
- Epochs: 3000
- Steps per example: 1

### Key Properties
- Binary (1-bit) state per cell
- Periodic boundaries (toroidal)
- Must learn all 512 possible 3×3 configurations

---

## Experiment 2: Checkerboard (Synchronous)

**Goal**: Generate checkerboard pattern from seed over 20 steps

### Architecture
- **Perception**: 16 kernels, `[9→8→4→2]` (no final reduction)
- **Update**: 17 layers `[513→256×10→128→64→32→16→8→8]`

### Training
- Learning rate: 0.05
- Batch size: 2
- Epochs: 500
- Steps: 20

### Key Properties
- 8-bit state per cell (multi-channel)
- Non-periodic boundaries
- Demonstrates boundary-size-invariant generalization (trains 16×16, works on 64×64)

---

## Experiment 3: Checkerboard (Asynchronous)

**Goal**: Same as sync, but with fault-tolerant async updates

### Architecture
- Same perception as sync
- **Update**: 21 layers (deeper for robustness)

### Training
- Batch size: 1
- Epochs: 800
- Steps: 50 (more steps for async convergence)
- **Async training**: fire_rate = 0.6

### Key Properties
- Demonstrates self-healing behavior
- Only ~60% of cells update per step (stochastic masking)
- Learned circuit is robust to cell failures

---

## Experiment 4: Growing Lizard

**Goal**: Grow complex lizard emoji pattern from seed

### Architecture
- **Perception**: 4 kernels (fewer than GoL), `[9→8→4→2→1]`
- **Update**: 11 layers `[513→512×8→256→128]`

### Training
- Learning rate: 0.06 (slightly higher)
- Batch size: 1
- Epochs: 3500
- Steps: 12

### Key Properties
- 128-bit state (richest representation)
- Periodic boundaries
- Generalizes from 20×20 training to 40×40 test grid
- Does NOT exploit boundary conditions

---

## Experiment 5: Colored G

**Goal**: Generate colored "G" pattern with 8-color RGB palette

### Architecture
- **Perception**: 4 kernels, `[9→8→4→2]`
- **Update**: 12 layers `[257→512×8→256→128→64]`
- **Total active gates**: 927 (most complex)

### Training
- Epochs: ~5000 (most training)
- Steps: 15

### Key Properties
- 64-bit state (RGB channels)
- 8-color palette using binary RGB
- Most complex circuit in the paper
- Required extensive hyperparameter tuning

---

## Common Architecture Patterns

### Perception Module
```
Input: 9 cells (3×3 neighborhood) × C channels = 9C bits
       ↓
Layer 1: "first_kernel" - 8 gates comparing center to each neighbor
       ↓
Layers 2+: "unique" connections - information mixing
       ↓
Output: 1 bit per kernel (or multi-bit for larger experiments)
```

### Update Module
```
Input: center_cell (C bits) + kernel_outputs (K × output_bits)
       ↓
Many "unique" layers (128-512 gates per layer)
       ↓
Reduction layers (halving until output size)
       ↓
Output: C bits (next cell state)
```

### Connection Topologies
- **first_kernel**: Layer 1 of perception - center vs each of 8 neighbors
- **unique**: All other layers - ensures each gate gets different input pair
- **random**: Not used in paper experiments (unique is better)

---

## Training Parameters (Common)

- **Optimizer**: AdamW (weight_decay=1e-2, b1=0.9, b2=0.99)
- **Gradient clipping**: 100.0
- **Pass-through init**: Gate index 3 = 10.0 logit
- **Fire rate** (async only): 0.6

---

## Implementation Notes for Rust Port

### Critical Design Decisions

1. **N-bit cells from start**: Design `Grid<N>` to support 1-128 bit states
2. **Channel-aware perception**: Each kernel processes C channels
3. **Flexible update input**: `center_channels + kernels × output_channels`
4. **Soft/hard mode**: Train with softmax, evaluate with argmax

### Suggested Rust Types

```rust
// Generic over number of channels
struct Grid<const C: usize> {
    width: usize,
    height: usize,
    cells: Vec<[f64; C]>,  // C channels per cell
}

// Or dynamic channels
struct DynamicGrid {
    width: usize,
    height: usize,
    channels: usize,
    cells: Vec<f64>,  // Flat: width * height * channels
}
```

### Phase Priority

1. GoL (C=1) - simplest, validates core architecture
2. Checkerboard (C=8) - first multi-channel test
3. Lizard/Colored G (C=64-128) - full capability demonstration
