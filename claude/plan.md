# Logicars Development Roadmap

## Primary References

1. **Paper**: [Differentiable Logic Cellular Automata](https://google-research.github.io/self-organising-systems/difflogic-ca/)
2. **Reference Implementation**: [Google Colab Notebook](https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/diffLogic_CA.ipynb) - JAX/Python implementation with critical details on architecture, training tricks, and hyperparameters
3. **Game of Life Rules**: Cell survives with 2-3 neighbors, dead cell born with exactly 3

---

## Macro-Plan: Barebones MVP → Full-Featured Crate

### **Phase 0: Foundation & Verification** (MVP Prerequisites)
**Goal**: Establish rock-solid primitives before touching CA

**Reference Implementation Insights**:
- Soft decoding: `softmax(weights, axis=-1)` during training
- Hard decoding: `one_hot(argmax(weights))` during inference
- All 16 gate outputs computed, then weighted sum by probabilities
- Critical: Pass-through gate (index 3) initialized to 10.0 for stability
- AdamW optimizer with gradient clipping at 100.0

#### **0.1 Single Gate Training**
- Implement one probabilistic logic gate with 16 binary operations
- Train to learn AND, OR, XOR individually
- **Verify gradient computation**: Numerical gradient checking against analytical
- **Verify convergence**: Loss → 0, probability distribution sharpens
- **Implementation notes**:
  - Use reference impl's soft/hard decoding approach
  - Test gradient clipping early (they use 100.0)
  - Consider deterministic ops for reproducibility (JAX uses `XLA_FLAGS='--xla_gpu_deterministic_ops=true'`)
- **Exit criteria**: >99% accuracy on truth tables, reproducible convergence

#### **0.2 Gate Layer**
- Multiple independent gates learning different operations simultaneously
- Verify no gradient interference between gates
- Test pass-through gate initialization trick
- **Exit criteria**: Can learn arbitrary boolean function combinations

#### **0.3 Multi-Layer Circuits**
- Stack gate layers to learn functions requiring depth (XOR from ANDs/ORs)
- Verify backpropagation through multiple layers
- **Reference impl detail**: Update module uses 128-512 hidden units across layers
- **Exit criteria**: Learn 2-3 layer circuits reliably

---

### **Phase 1: Game of Life MVP**
**Goal**: Minimal working CA that learns Conway's Game of Life

**Reference Implementation Architecture**:
- **Perception Module**: 4-16 parallel kernels on 3x3 neighborhoods
  - First layer: "first_kernel" topology (mimics cell interactions)
  - Subsequent layers: "unique" connections (each gate gets different inputs)
  - Outputs concatenated with original center cell value
- **Update Module**: Deep network (128-512 hidden units) → next state
- **Loss**: Mean squared error `sum((output - target)²)`
- **Training**: AdamW, LR ~0.05-0.06, batch sizes 1-20
- **Fire rate**: 0.6 for asynchronous training (enables fault tolerance)

#### **1.1 Perception Circuit**
- 3x3 neighborhood → single bit output
- Start with 4 parallel kernels (reference impl minimum)
- Implement "first_kernel" topology from reference
- Train on known Game of Life patterns
- **Exit criteria**: >95% accuracy on all 512 neighborhood configurations

#### **1.2 Update Circuit Integration**
- Implement concatenation: perception outputs + center cell value
- Deep network following reference architecture (start with 128 hidden units)
- Full forward pass on static grids
- **Exit criteria**: Correct next-state prediction on test grids

#### **1.3 Training Loop**
- Generate Game of Life training data (random initial states + correct next states)
- Implement MSE loss matching reference
- AdamW optimizer with gradient clipping (100.0)
- Learning rate ~0.05-0.06 as baseline
- **Exit criteria**: Converges on Game of Life, >95% hard accuracy, stable loss

#### **1.4 Multi-Step Rollout**
- Apply learned rule iteratively (t → t+1 → t+2...)
- Test on classic patterns (gliders, blinkers, still lifes)
- Optional: Implement asynchronous updates (fire rate 0.6) for robustness testing
- **Exit criteria**: Stable multi-step simulation matching true Game of Life

---

### **Phase 2: Library Foundations**
**Goal**: Transform working code into reusable crate

#### **2.1 API Design**
- Separate core logic from Game of Life specifics
- Clean abstractions: `PerceptionCircuit`, `UpdateCircuit`, `CA`, `Trainer`
- Builder pattern for configuration
- **Key parameters to expose**:
  - Number of perception kernels (4-16)
  - Update module hidden units
  - Training hyperparameters (LR, clipping, fire rate)
  - Gate initialization strategies
- **Exit criteria**: Can instantiate and train without touching internals

#### **2.2 Serialization & Checkpointing**
- Save/load trained circuits (gate probability distributions)
- Export to interpretable formats (circuit diagrams, truth tables)
- Store both soft weights and hard-decoded circuits
- **Exit criteria**: Can serialize, load, and resume training

#### **2.3 Testing Infrastructure**
- Property-based tests for gates (commutativity, identities)
- Integration tests for training convergence
- Regression tests on Game of Life (compare to reference impl results)
- Numerical gradient checking tests
- **Exit criteria**: Comprehensive test suite, CI passing

#### **2.4 Documentation**
- API docs with examples
- Tutorial: "Train your first CA"
- Architecture explanation (with diagrams comparing to reference)
- Document training tricks (gate initialization, clipping, fire rate)
- **Exit criteria**: New user can train Game of Life from docs alone

---

### **Phase 3: Generalization to Other CA Rules**
**Goal**: Validate that library works beyond Game of Life

#### **3.1 Parameterize CA Rules**
- Support different neighborhood types (Moore, von Neumann)
- Configurable state spaces (binary → multi-bit)
- Flexible perception kernel counts and topologies
- **Exit criteria**: Can define arbitrary CA rule learning tasks

#### **3.2 Validate on Known Rules**
- Learn Wolfram elementary CA (Rule 30, Rule 110)
- Learn Brian's Brain, WireWorld, Seeds
- Compare learned circuits to ground truth
- **Reference impl tested**: Checkerboard patterns, emoji growth
- **Exit criteria**: ≥3 different CA rules learned successfully

#### **3.3 Benchmarking Suite**
- Standardized tasks with difficulty ratings
- Performance metrics (convergence speed, final accuracy, circuit size)
- Compare Rust impl performance to reference Python/JAX
- **Exit criteria**: Reproducible benchmark results

---

### **Phase 4: Advanced Features**
**Goal**: Push beyond paper's scope, explore what's possible

#### **4.1 Architecture Search**
- Hyperparameter tuning (# perception circuits, layer depth)
- Automatically find minimal circuit for given accuracy
- Explore different connection topologies beyond "first_kernel" and "unique"
- **Exit criteria**: Can recommend architecture for new CA learning tasks

#### **4.2 Multi-State CA**
- Extend from binary to multi-valued states (4-state, 8-state)
- Use soft gates with more inputs or hierarchical circuits
- **Exit criteria**: Learn at least one non-binary CA rule

#### **4.3 Larger Neighborhoods**
- Beyond 3x3 (5x5, 7x7)
- Efficient handling of exponentially growing input space
- **Exit criteria**: Learn rule requiring larger context

#### **4.4 Inverse Problems**
- Given desired behavior (pattern generation), find CA rule
- Optimize for specific properties (entropy, stability, fault tolerance)
- Reference impl shows self-healing with async updates - formalize this
- **Exit criteria**: One working example of behavior-driven rule discovery

---

### **Phase 5: Ecosystem & Interop**
**Goal**: Make library useful in broader contexts

#### **5.1 Python Bindings (PyO3)**
- Expose core API to Python
- Jupyter notebook examples (similar format to reference Colab)
- Integration with PyTorch/JAX for hybrid models
- **Exit criteria**: `pip install logicars`, run training from Python

#### **5.2 Visualization Tools**
- Render CA evolution (grids → animations)
- Visualize learned circuits (gate diagrams)
- Training progress dashboards
- Reproduce reference impl's visualizations in Rust
- **Exit criteria**: Publication-quality figures from library

#### **5.3 R Bindings (extendr)**
- Statistical analysis of learned rules
- Integration with R's visualization ecosystem
- **Exit criteria**: CRAN-ready package (if desired)

#### **5.4 WASM/Browser Demo**
- Compile to WebAssembly
- Interactive demo: train CA in browser
- **Exit criteria**: Shareable web demo of library capabilities

---

### **Phase 6: Research Extensions** (Optional)
**Goal**: Novel contributions beyond replication

#### **6.1 Transfer Learning**
- Train on Game of Life, fine-tune on similar rules
- Investigate what circuits learn about CA structure

#### **6.2 Probabilistic/Stochastic CA**
- Rules with randomness
- Soft execution during inference (not just training)
- Extend reference impl's fire rate concept to probabilistic rules

#### **6.3 Continuous-Space CA**
- Apply to neural CA (continuous states)
- Hybrid discrete logic + continuous functions

#### **6.4 Hardware Synthesis**
- Export learned circuits to Verilog/VHDL
- FPGA deployment of learned rules
- Hard-decoded circuits are hardware-ready by design

---

## Critical Success Factors

1. **Verification First**: Never proceed to next phase with failing tests
2. **Minimal Viable Increments**: Each sub-phase should take days, not weeks
3. **Empirical Validation**: Always compare to reference implementation results when available
4. **Fail Fast**: If convergence fails, debug at current phase before adding complexity
5. **Documentation as You Go**: Write docs when API is fresh in mind

---

## Risk Mitigation

### Convergence Issues
- Keep detailed training logs, visualize gradients
- Compare to numerical gradients (automated tests)
- Reference impl needed deterministic ops for reproducibility - consider early
- Use reference hyperparameters as baseline (LR 0.05-0.06, clipping 100.0)

### Architecture Mistakes
- Validate each layer independently before integration
- Compare intermediate outputs to reference impl when possible
- Test pass-through gate initialization (index 3 = 10.0)

### Over-Engineering
- Resist abstractions until pattern emerges from 2-3 concrete examples
- Reference impl is ~300 lines - start similarly minimal

### Scope Creep
- Phase 1 delivers value (working Game of Life trainer)
- Everything after is enhancement
- Reference impl proves core concept works - focus on correct replication first

---

## Key Implementation Tricks from Reference

These should be incorporated throughout development:

1. **Gate initialization**: Pass-through gate (index 3) = 10.0 for training stability
2. **Gradient clipping**: 100.0 (prevents explosion in discrete optimization)
3. **Soft/Hard decoding**: Train soft, evaluate hard, report both metrics
4. **Concatenation**: Always include center cell value in update module inputs
5. **Fire rate**: 0.6 for async training enables fault tolerance and generalization
6. **Deterministic ops**: Critical for reproducible results (worth performance cost during development)
7. **Perception topologies**: "first_kernel" for layer 1, "unique" for subsequent layers
8. **Batch diversity**: Random sampling from full configuration space, not sequential

---

This plan takes you from "can we even train one gate" to "fully-featured library with bindings and novel applications." The key insight from Claude.md is that **previous attempts failed by skipping validation steps**. The reference implementation provides ground truth for every phase - use it liberally to validate correctness before optimizing or extending.
