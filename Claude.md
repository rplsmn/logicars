# Claude.md - Logicars Project Guide

## Project Goal

Replicate the results from Google Research's **Differentiable Logic Cellular Automata** paper:
https://google-research.github.io/self-organising-systems/difflogic-ca/

The paper demonstrates learning cellular automata rules (like Conway's Game of Life) using differentiable logic gate networks - discrete binary circuits that can be trained via gradient descent.

---

## Technical Domain Knowledge Required

### Cellular Automata (CA)
- 2D grids where each cell has a state (binary in this case)
- Cells update based on their neighborhood (Moore neighborhood = 8 neighbors + self)
- Rules are applied uniformly and synchronously across the grid
- Conway's Game of Life: cell survives with 2-3 neighbors, dead cell becomes alive with exactly 3 neighbors

### Differentiable Logic Gate Networks (DLGNs)
- 16 possible binary operations between two inputs: FALSE, AND, A_AND_NOT_B, A, NOT_A_AND_B, B, XOR, OR, NOR, XNOR, NOT_B, A_OR_NOT_B, NOT_A, NOT_A_OR_B, NAND, TRUE
- **Key insight**: During training, each gate maintains a probability distribution over all 16 operations
- **Soft computation**: Output = weighted sum of all gate outputs by their probabilities
- **Hard computation**: Use only the highest-probability gate (for inference)
- Gradients flow through the soft computation to update gate probabilities

### Architecture (per the paper)
1. **Perception circuits**: Process the 9-cell neighborhood
   - Input: 9 cells × state_size bits = 9 × 1 = 9 bits for Game of Life
   - Architecture: 9 → 8 → 4 → 2 → 1 (pyramid reduction)
   - Multiple perception circuits run in parallel (paper uses ~16)

2. **Update circuit**: Computes next state from perception outputs + current state
   - Input: n_perception_outputs + state_size
   - Deeper network with many layers (paper suggests 8-16 layers of 128-512 gates)
   - Output: state_size bits (1 for Game of Life)

### Training Methodology
- **Loss**: Mean squared error between soft predictions and targets
- **Gradient flow**: Through update circuit → perception circuits → gate probabilities
- **Temperature**: Controls softmax sharpness (high = exploration, low = exploitation)
- **Key**: Gradients update the probability distributions, not the gate types directly

---

## Rust & PyO3 Expertise

This project uses:
- **Rust** for the core CA and circuit implementation (performance-critical)
- **PyO3** for Python bindings (training orchestration)
- **ndarray** for multi-dimensional arrays
- **rayon** for parallelism

### Important Rust Patterns
- Ownership and borrowing must be correct, especially in training loops
- Use `Arc<Mutex<>>` sparingly and only when truly needed
- Prefer functional iteration over manual loops where it improves clarity
- Keep unsafe code to zero unless absolutely necessary

---

## Development Principles

### 1. Working Code First
- Every change must compile and run
- No placeholder code, no `todo!()`, no `unimplemented!()`
- No "we'll add this later" comments
- If a feature isn't needed yet, don't add the scaffolding

### 2. Incremental Verification
- Test each component in isolation before integration
- For a CA: verify neighborhood extraction, then single gate, then circuit, then full step
- Use known patterns (blinker, glider) as smoke tests
- Print intermediate values during debugging, remove when done

### 3. Follow the Paper Exactly First
- Implement what the paper describes, not optimizations
- Verify results match before adding improvements
- The paper's hyperparameters are a starting point, not gospel

### 4. Clear Abstractions
- `LogicGate`: single gate with soft/hard compute and probability updates
- `GateLayer`: parallel gates with same input size
- `Circuit`: stack of layers
- `PerceptionCircuit`: neighborhood → single output
- `UpdateCircuit`: perceptions + state → next state
- `DiffLogicCA`: grid + circuits + training loop

### 5. Debugging Strategy
When training doesn't converge:
1. Verify forward pass is correct (hard computation matches expected Game of Life)
2. Check gradient magnitudes (should be non-zero, not exploding)
3. Verify probability updates move in the right direction
4. Start with simpler targets (always-true, always-false, simple patterns)
5. Reduce architecture complexity to isolate issues

---

## Current State Assessment

The existing code has:
- Basic structure for gates, circuits, perception, and update
- PyO3 bindings for Python training
- Training loop with soft/hard loss computation

Potential issues to investigate:
- Gradient computation correctness
- Probability update mechanism
- Architecture sizing (may be over/under-parameterized)
- Temperature scheduling
- Loss function (center-cell-only vs full-grid)

---

## Step-by-Step Replication Plan

### Phase 1: Verification
1. Verify Game of Life rules are correctly implemented (ground truth)
2. Verify hard forward pass produces valid CA steps
3. Create minimal test cases with known inputs/outputs

### Phase 2: Single Gate Training
1. Train a single gate to learn AND, OR, XOR
2. Verify gradients flow correctly
3. Verify probability distributions converge to correct gate

### Phase 3: Simple Circuit Training
1. Train a 2-input → 1-output circuit
2. Learn XOR (requires multi-layer)
3. Verify gradient backpropagation through layers

### Phase 4: Full Game of Life
1. Use paper's architecture
2. Start with subset of training data
3. Track loss curves and gate probability evolution
4. Tune hyperparameters systematically

---

## Success Criteria

The replication is successful when:
1. Training loss decreases consistently
2. Hard accuracy reaches >95% on all 512 Game of Life configurations
3. Learned rules generalize to larger grids
4. Gate distributions stabilize (low entropy, clear winners)

---

## Reference Implementation Notes

From the paper:
- Game of Life learned with 336 active gates
- Perception kernels: 8→4→2→1 (channel-wise processing)
- Update networks: 8-16 layers, 128-512 gates per layer
- Training used squared difference loss
- Asynchronous updates improved robustness

---

## Commands

```bash
# Build the Rust library
maturin develop --release

# Run training
python train_optimized.py --epochs 200 --lr 0.01 --tmp 2 --batchsize 128

# Run tests
cargo test
```
