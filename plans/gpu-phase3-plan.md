# GPU Phase 3: Backward Pass on GPU

## Overview

**Goal**: Implement GPU compute shaders for backward pass (gradient computation) for gate layers. Enable full BPTT training loop on GPU.

**Estimated Duration**: 3-4 days

**Dependencies**: Phase 2 complete (forward pass on GPU working and verified)

---

## Success Criteria

1. ✅ Gate backward shader computes dL/dlogits correctly
2. ✅ Gate backward shader computes dL/dinputs correctly
3. ✅ Layer backward matches CPU gradients within 1e-4
4. ✅ Full BPTT backward pass matches CPU within 1e-4
5. ✅ Multi-step rollout + backward works correctly
6. ✅ Gradient accumulation across steps works

---

## Background: Gradient Computation

### Single Gate Backward Pass

For a probabilistic gate with softmax over 16 operations:

```
Forward:
  probs[i] = exp(logits[i] - max) / sum(exp(logits - max))
  output = sum(probs[i] * op[i](a, b))

Backward (given dL/d_output):
  
  // Gradients w.r.t. logits (for learning)
  for i in 0..16:
    dL/d_logits[i] = dL/d_output * d_output/d_logits[i]
    
    // d_output/d_logits[i] = op[i](a,b) * d_probs[i]/d_logits[i]
    //                      = op[i](a,b) * probs[i] * (1 - probs[i])  (if i == j)
    //                      = op[i](a,b) * probs[i] * (-probs[j])     (if i != j)
    
    // Simplified: dL/d_logits[i] = dL/d_output * probs[i] * (op[i] - output)
  
  // Gradients w.r.t. inputs (for backprop to earlier layers)
  dL/d_a = dL/d_output * sum(probs[i] * d_op[i]/d_a)
  dL/d_b = dL/d_output * sum(probs[i] * d_op[i]/d_b)
```

### Derivatives of Binary Operations

| Op | Index | Formula | d/da | d/db |
|----|-------|---------|------|------|
| False | 0 | 0 | 0 | 0 |
| And | 1 | a*b | b | a |
| A∧¬B | 2 | a*(1-b) | 1-b | -a |
| A | 3 | a | 1 | 0 |
| ¬A∧B | 4 | (1-a)*b | -b | 1-a |
| B | 5 | b | 0 | 1 |
| Xor | 6 | a+b-2ab | 1-2b | 1-2a |
| Or | 7 | a+b-ab | 1-b | 1-a |
| Nor | 8 | 1-a-b+ab | b-1 | a-1 |
| Xnor | 9 | 1-a-b+2ab | 2b-1 | 2a-1 |
| ¬B | 10 | 1-b | 0 | -1 |
| B→A | 11 | 1-(1-a)*b | b | a-1 |
| ¬A | 12 | 1-a | -1 | 0 |
| A→B | 13 | 1-a*(1-b) | b-1 | a |
| Nand | 14 | 1-ab | -b | -a |
| True | 15 | 1 | 0 | 0 |

---

## Task Breakdown

### Task 3.1: Implement Gate Backward Shader

**Description**: WGSL shader for computing gradients through a gate layer

**Implementation** (`src/gpu/shaders/gate_backward.wgsl`):

```wgsl
struct LayerConfig {
    num_gates: u32,
    input_offset: u32,
    output_offset: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> config: LayerConfig;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> logits: array<f32>;
@group(0) @binding(3) var<storage, read> wires: array<u32>;
@group(0) @binding(4) var<storage, read> outputs: array<f32>;  // Forward pass outputs (needed for gradient)
@group(0) @binding(5) var<storage, read> d_outputs: array<f32>;  // Gradient from next layer
@group(0) @binding(6) var<storage, read_write> d_logits: array<f32>;  // Output: gradient w.r.t. logits
@group(0) @binding(7) var<storage, read_write> d_inputs: array<f32>;  // Output: gradient w.r.t. inputs

// Derivative of binary_op w.r.t. input a
fn d_op_da(op: u32, a: f32, b: f32) -> f32 {
    switch (op) {
        case 0u: { return 0.0; }           // False
        case 1u: { return b; }             // And: d(a*b)/da = b
        case 2u: { return 1.0 - b; }       // A∧¬B: d(a*(1-b))/da = 1-b
        case 3u: { return 1.0; }           // A: d(a)/da = 1
        case 4u: { return -b; }            // ¬A∧B: d((1-a)*b)/da = -b
        case 5u: { return 0.0; }           // B
        case 6u: { return 1.0 - 2.0 * b; } // Xor
        case 7u: { return 1.0 - b; }       // Or
        case 8u: { return b - 1.0; }       // Nor
        case 9u: { return 2.0 * b - 1.0; } // Xnor
        case 10u: { return 0.0; }          // ¬B
        case 11u: { return b; }            // B→A: d(1-(1-a)*b)/da = b
        case 12u: { return -1.0; }         // ¬A
        case 13u: { return b - 1.0; }      // A→B: d(1-a*(1-b))/da = b-1
        case 14u: { return -b; }           // Nand
        default: { return 0.0; }           // True
    }
}

// Derivative of binary_op w.r.t. input b
fn d_op_db(op: u32, a: f32, b: f32) -> f32 {
    switch (op) {
        case 0u: { return 0.0; }           // False
        case 1u: { return a; }             // And
        case 2u: { return -a; }            // A∧¬B
        case 3u: { return 0.0; }           // A
        case 4u: { return 1.0 - a; }       // ¬A∧B
        case 5u: { return 1.0; }           // B
        case 6u: { return 1.0 - 2.0 * a; } // Xor
        case 7u: { return 1.0 - a; }       // Or
        case 8u: { return a - 1.0; }       // Nor
        case 9u: { return 2.0 * a - 1.0; } // Xnor
        case 10u: { return -1.0; }         // ¬B
        case 11u: { return a - 1.0; }      // B→A
        case 12u: { return 0.0; }          // ¬A
        case 13u: { return a; }            // A→B
        case 14u: { return -a; }           // Nand
        default: { return 0.0; }           // True
    }
}

fn binary_op(op: u32, a: f32, b: f32) -> f32 {
    // Same as forward shader - copy from gate_forward.wgsl
    // ...
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gate_idx = gid.x;
    if (gate_idx >= config.num_gates) {
        return;
    }
    
    // Read inputs
    let wire_a = wires[gate_idx * 2u];
    let wire_b = wires[gate_idx * 2u + 1u];
    let a = inputs[config.input_offset + wire_a];
    let b = inputs[config.input_offset + wire_b];
    
    // Read upstream gradient
    let d_out = d_outputs[config.output_offset + gate_idx];
    
    // Recompute softmax (or could cache from forward pass)
    var max_logit = logits[gate_idx * 16u];
    for (var i = 1u; i < 16u; i++) {
        max_logit = max(max_logit, logits[gate_idx * 16u + i]);
    }
    
    var exp_sum = 0.0;
    var probs: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        probs[i] = exp(logits[gate_idx * 16u + i] - max_logit);
        exp_sum += probs[i];
    }
    for (var i = 0u; i < 16u; i++) {
        probs[i] /= exp_sum;
    }
    
    // Compute output (needed for softmax gradient)
    var output = 0.0;
    for (var i = 0u; i < 16u; i++) {
        output += probs[i] * binary_op(i, a, b);
    }
    
    // Gradient w.r.t. logits: dL/dlogits[i] = dL/dout * probs[i] * (op[i] - output)
    for (var i = 0u; i < 16u; i++) {
        let op_val = binary_op(i, a, b);
        d_logits[gate_idx * 16u + i] = d_out * probs[i] * (op_val - output);
    }
    
    // Gradient w.r.t. inputs: accumulate across operations
    var d_a = 0.0;
    var d_b = 0.0;
    for (var i = 0u; i < 16u; i++) {
        d_a += probs[i] * d_op_da(i, a, b);
        d_b += probs[i] * d_op_db(i, a, b);
    }
    d_a *= d_out;
    d_b *= d_out;
    
    // Atomic add to handle multiple gates writing to same input
    // Note: wgpu doesn't have atomicAdd for f32, need to use atomicAdd on i32 or restructure
    // For now, assume each input is read by at most one gate per layer (true for our architecture)
    d_inputs[config.input_offset + wire_a] += d_a;
    d_inputs[config.input_offset + wire_b] += d_b;
}
```

**Issue: Atomic Float Addition**

WGSL doesn't support `atomicAdd` for floats. Options:
1. **Scatter-gather pattern**: Each gate writes to separate slots, then reduce
2. **Int32 atomics**: Convert floats to fixed-point, use atomicAdd
3. **Separate kernel**: Compute d_inputs in a separate reduction kernel

**Recommended**: Use scatter-gather pattern:
```wgsl
// Each gate writes its contributions to a "contributions" buffer
// contributions[gate_idx * 2 + 0] = (wire_a, d_a)
// contributions[gate_idx * 2 + 1] = (wire_b, d_b)
// Then a separate kernel gathers: d_inputs[wire] = sum over gates that use wire
```

**Tests**:
```rust
#[test]
fn test_gate_backward_shader_logit_gradients() {
    let ctx = GpuContext::new().unwrap();
    
    // Single gate, known configuration
    let mut logits = vec![0.0f32; 16];
    logits[3] = 10.0; // Pass-through
    
    let inputs = vec![0.7f32, 0.3f32];
    let wires = vec![0u32, 1u32];
    let d_output = vec![1.0f32]; // Gradient of 1.0 from loss
    
    let (d_logits, d_inputs) = ctx.run_gate_backward(
        &inputs, &logits, &wires, &d_output
    ).unwrap();
    
    // Compare with CPU
    let cpu_gate = ProbabilisticGate::from_logits(logits.iter().map(|&x| x as f64).collect(), 0, 1);
    let (cpu_d_logits, cpu_d_a, cpu_d_b) = cpu_gate.backward(0.7, 0.3, 1.0);
    
    // Logit gradients should match
    for i in 0..16 {
        let diff = (d_logits[i] as f64 - cpu_d_logits[i]).abs();
        assert!(diff < 1e-4, "Logit {} gradient: GPU={}, CPU={}", i, d_logits[i], cpu_d_logits[i]);
    }
}

#[test]
fn test_gate_backward_shader_input_gradients() {
    // Similar test for d_inputs
}
```

**Exit Criteria**:
- [ ] Shader compiles
- [ ] Logit gradients match CPU within 1e-4
- [ ] Input gradients match CPU within 1e-4
- [ ] Multiple gates don't corrupt shared inputs

---

### Task 3.2: Handle Gradient Accumulation for Shared Inputs

**Description**: Implement correct gradient accumulation when multiple gates read from the same input

**Design**:

Our architecture uses `unique` connections where each pair of inputs connects to exactly one gate. However, we still need correct accumulation for:
1. Different layers sharing intermediate activations
2. BPTT where the same parameters are used at multiple timesteps

**Implementation - Scatter-Gather Pattern**:

```rust
// In GPU backward pass:
struct GradientContribution {
    wire_idx: u32,
    gradient: f32,
}

// Kernel 1: Compute contributions (one per gate output)
// contributions: [num_gates × 2] - each gate contributes to 2 inputs
fn compute_gradient_contributions(...) {
    // Each gate writes its d_a and d_b with wire indices
}

// Kernel 2: Gather contributions into d_inputs
@compute @workgroup_size(256)
fn gather_gradients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let input_idx = gid.x;
    if (input_idx >= num_inputs) { return; }
    
    var total = 0.0;
    for (var g = 0u; g < num_gates; g++) {
        if (wires[g * 2u + 0u] == input_idx) {
            total += contributions[g * 2u + 0u];
        }
        if (wires[g * 2u + 1u] == input_idx) {
            total += contributions[g * 2u + 1u];
        }
    }
    d_inputs[input_idx] = total;
}
```

**Alternative: Sorted Wire Indices**

Pre-sort gates by wire indices for coalesced access:
```rust
// Precompute: for each input, list of (gate_idx, which_wire) that read it
struct WireConsumer {
    input_idx: u32,
    gate_idx: u32,
    is_wire_b: bool,
}
// Sort by input_idx, then gather is O(consumers_per_input)
```

**Tests**:
```rust
#[test]
fn test_gradient_accumulation_shared_inputs() {
    // Create layer where multiple gates read same input
    // Verify gradients accumulate correctly
}
```

**Exit Criteria**:
- [ ] Scatter-gather pattern implemented
- [ ] Gradients accumulate correctly for shared inputs
- [ ] Performance acceptable (no O(n²) loops)

---

### Task 3.3: Implement Layer Backward on GPU

**Description**: Complete backward pass for a GateLayer

**Implementation** (`src/gpu/gate_layer.rs`):

```rust
impl GpuContext {
    /// Backward pass through a gate layer
    /// 
    /// # Arguments
    /// * `layer` - The gate layer (for logits and wires)
    /// * `inputs` - Forward pass inputs (cached)
    /// * `d_outputs` - Gradient of loss w.r.t. layer outputs
    /// 
    /// # Returns
    /// * `d_logits` - Gradient w.r.t. gate logits [num_gates × 16]
    /// * `d_inputs` - Gradient w.r.t. layer inputs
    pub fn run_gate_layer_backward(
        &self,
        layer: &GateLayer,
        inputs: &[f64],
        d_outputs: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        // Convert to f32
        let inputs_f32: Vec<f32> = inputs.iter().map(|&x| x as f32).collect();
        let d_outputs_f32: Vec<f32> = d_outputs.iter().map(|&x| x as f32).collect();
        let logits_f32 = layer.get_logits_flat_f32();
        let wires = layer.get_wires_flat();
        
        // Create buffers
        let d_logits_f32 = vec![0.0f32; layer.num_gates() * 16];
        let d_inputs_f32 = vec![0.0f32; inputs.len()];
        
        // Run backward kernels
        // ... (buffer creation, dispatch, readback)
        
        // Convert back to f64
        let d_logits: Vec<f64> = d_logits_f32.iter().map(|&x| x as f64).collect();
        let d_inputs: Vec<f64> = d_inputs_f32.iter().map(|&x| x as f64).collect();
        
        Ok((d_logits, d_inputs))
    }
}
```

**Tests**:
```rust
#[test]
fn test_gate_layer_backward_vs_cpu() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(42);
    
    for (input_size, output_size) in [(8, 4), (16, 8), (64, 32)] {
        let layer = GateLayer::new(input_size, output_size, ConnectionType::Unique, &mut rng);
        let inputs: Vec<f64> = (0..input_size).map(|_| rng.next_f64()).collect();
        let d_outputs: Vec<f64> = (0..output_size).map(|_| rng.next_f64() - 0.5).collect();
        
        // CPU backward
        let cpu_result = layer.backward(&inputs, &d_outputs);
        
        // GPU backward
        let (gpu_d_logits, gpu_d_inputs) = ctx.run_gate_layer_backward(
            &layer, &inputs, &d_outputs
        ).unwrap();
        
        // Compare d_inputs
        let max_diff_inputs: f64 = cpu_result.d_inputs.iter()
            .zip(gpu_d_inputs.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0, f64::max);
        assert!(max_diff_inputs < 1e-4, "d_inputs max diff = {}", max_diff_inputs);
        
        // Compare d_logits
        let cpu_d_logits_flat = cpu_result.d_logits_flat();
        let max_diff_logits: f64 = cpu_d_logits_flat.iter()
            .zip(gpu_d_logits.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0, f64::max);
        assert!(max_diff_logits < 1e-4, "d_logits max diff = {}", max_diff_logits);
    }
}
```

**Exit Criteria**:
- [ ] `run_gate_layer_backward` implemented
- [ ] Input gradients match CPU within 1e-4
- [ ] Logit gradients match CPU within 1e-4

---

### Task 3.4: Implement BPTT on GPU

**Description**: Backpropagation through time for multi-step rollout

**Design**:

```
Forward pass (T steps):
  grid_0 → step → grid_1 → step → grid_2 → ... → grid_T

Backward pass:
  d_loss/d_grid_T → backward_step → d_loss/d_grid_{T-1} → ... → d_loss/d_grid_0
  
  At each step, accumulate d_logits for all layers
```

**Implementation** (`src/gpu/training.rs`):

```rust
impl GpuContext {
    /// Run BPTT on GPU
    /// 
    /// # Returns
    /// Total gradient w.r.t. all model parameters
    pub fn backward_through_time(
        &self,
        model: &DiffLogicCA,
        grids: &[NGrid],  // All intermediate grids from forward pass
        target: &NGrid,
        loss_channel: Option<usize>,
    ) -> Result<ModelGradients, GpuError> {
        let num_steps = grids.len() - 1;
        let final_grid = &grids[num_steps];
        
        // Initialize gradients
        let mut total_gradients = ModelGradients::zeros_like(model);
        
        // Compute loss gradient w.r.t. final grid
        let mut d_grid = compute_loss_gradient(final_grid, target, loss_channel);
        
        // Backward through each step
        for step in (0..num_steps).rev() {
            let input_grid = &grids[step];
            
            // Backward through all cells
            let (step_gradients, d_grid_prev) = self.backward_ca_step(
                model, input_grid, &d_grid
            )?;
            
            // Accumulate gradients
            total_gradients.add_inplace(&step_gradients);
            
            // Propagate to previous step
            d_grid = d_grid_prev;
        }
        
        Ok(total_gradients)
    }
    
    /// Backward through a single CA step
    fn backward_ca_step(
        &self,
        model: &DiffLogicCA,
        input_grid: &NGrid,
        d_output_grid: &NGrid,
    ) -> Result<(ModelGradients, NGrid), GpuError> {
        // For each cell, backward through update then perception
        // Accumulate gradients across all cells
        
        // This should be batched like forward pass
        // ...
    }
}

struct ModelGradients {
    perception_gradients: Vec<Vec<f64>>,  // Per-layer, per-gate logit gradients
    update_gradients: Vec<Vec<f64>>,
}
```

**Key Consideration: Caching Forward Activations**

For backward pass, we need intermediate activations from forward pass. Options:
1. **Recompute**: Run forward again during backward (slower but less memory)
2. **Cache all**: Store all intermediate values (faster but more memory)
3. **Checkpointing**: Cache every N steps, recompute between (balanced)

For our problem size (16×16×8 grid, 3040 gates, 20 steps), caching all is feasible:
- Per-step cache: ~30KB per cell × 256 cells = ~8MB
- 20 steps: ~160MB total (easily fits in 16GB GPU memory)

**Tests**:
```rust
#[test]
fn test_bptt_gpu_vs_cpu() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(123);
    
    let model = create_small_checkerboard_model();
    let seed = create_random_seed(16, 8, &mut rng);
    let target = create_checkerboard(16, 2, 8);
    let num_steps = 5; // Fewer steps for faster test
    
    // Forward pass (same on CPU and GPU)
    let grids = run_forward_steps(&model, &seed, num_steps);
    
    // CPU backward
    let cpu_grads = backward_through_time_cpu(&model, &grids, &target, Some(0));
    
    // GPU backward
    let gpu_grads = ctx.backward_through_time(&model, &grids, &target, Some(0)).unwrap();
    
    // Compare gradients
    let max_diff = compare_gradients(&cpu_grads, &gpu_grads);
    assert!(max_diff < 1e-3, "Gradient max diff = {}", max_diff);
}
```

**Exit Criteria**:
- [ ] BPTT implemented on GPU
- [ ] Gradients match CPU within 1e-3
- [ ] 20-step rollout completes without memory issues

---

### Task 3.5: Implement Batched Backward Pass

**Description**: Batch backward computation across all cells (matching batched forward)

**Design**:

Just as forward batched all cells into single kernel dispatches, backward should too:

```
Instead of:
  for each cell:
    backward_perception(d_output_cell)
    backward_update(d_perception_output_cell)

Do:
  1. Compute d_output for all cells as contiguous buffer
  2. Dispatch update backward for ALL gates across ALL cells
  3. Dispatch perception backward for ALL gates across ALL cells
```

**Buffer Layout**:
```
d_update_inputs: [num_cells × perception_output_size]
                = [256 × 264] = 67,584 f32

d_perception_inputs: [num_cells × neighborhood_size]  
                   = [256 × 9 × 8] = 18,432 f32 (not directly used, but shows scale)
```

**Implementation**:

Similar to Task 2.7 but for backward pass. The shader needs:
```wgsl
@compute @workgroup_size(64, 4, 1)
fn backward_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gate_idx = gid.x;
    let cell_idx = gid.y;
    // ... backward computation with cell offsets
}
```

**Tests**:
```rust
#[test]
fn test_batched_backward_vs_sequential() {
    // Similar to test_batched_forward_vs_sequential
}

#[test]
fn test_batched_backward_performance() {
    // Verify speedup over sequential
}
```

**Exit Criteria**:
- [ ] Batched backward implemented
- [ ] Results match sequential backward
- [ ] Performance improvement measured

---

### Task 3.6: Implement Gradient Clipping on GPU

**Description**: Apply gradient clipping (max 100.0) on GPU

**Implementation**:

Simple shader that clamps gradients:

```wgsl
@group(0) @binding(0) var<storage, read_write> gradients: array<f32>;

@compute @workgroup_size(256)
fn clip_gradients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gradients)) { return; }
    
    let grad = gradients[idx];
    gradients[idx] = clamp(grad, -100.0, 100.0);
}
```

**Tests**:
```rust
#[test]
fn test_gradient_clipping_gpu() {
    let ctx = GpuContext::new().unwrap();
    
    let gradients = vec![-200.0, -50.0, 0.0, 50.0, 200.0];
    let clipped = ctx.clip_gradients(&gradients, 100.0).unwrap();
    
    assert_eq!(clipped, vec![-100.0, -50.0, 0.0, 50.0, 100.0]);
}
```

**Exit Criteria**:
- [ ] Gradient clipping shader works
- [ ] Values clamped correctly

---

## Final Checklist

| Task | Status | Verified By |
|------|--------|-------------|
| 3.1 Gate backward shader | ⬜ | `test_gate_backward_shader_*` |
| 3.2 Gradient accumulation | ⬜ | `test_gradient_accumulation_*` |
| 3.3 Layer backward | ⬜ | `test_gate_layer_backward_vs_cpu` |
| 3.4 BPTT on GPU | ⬜ | `test_bptt_gpu_vs_cpu` |
| 3.5 Batched backward | ⬜ | `test_batched_backward_*` |
| 3.6 Gradient clipping | ⬜ | `test_gradient_clipping_gpu` |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Atomic float issue | High | High | Use scatter-gather pattern |
| Gradient mismatch | High | High | Verify at each layer individually |
| Memory for caching | Medium | Medium | Use checkpointing if needed |
| Numerical instability | Medium | Medium | Use f32 with care, verify against CPU |

---

## Memory Budget (Checkerboard 16×16)

| Buffer | Size | Notes |
|--------|------|-------|
| Grid state | 2 KB | 256 cells × 8 channels |
| Perception activations | 67 KB | 256 cells × 264 features |
| Update activations | 656 KB | 256 cells × 2560 (10 layers × 256) |
| Gradient buffers | ~720 KB | Mirror of activations |
| Model parameters | ~770 KB | Logits for all gates |
| **Total per step** | ~2.2 MB | |
| **Total for 20 steps** | ~44 MB | Fits easily in 16GB |

---

## Notes for Implementation

1. **Verify at each layer**: Don't move to next task until gradients match
2. **Start without batching**: Get single-cell backward working first
3. **Watch for NaN**: Gradient computation can produce NaN with bad inputs
4. **Profile memory**: Ensure we don't exceed GPU memory with caching
5. **Consider recomputation**: If memory tight, recompute forward during backward

---

## Next Phase

After Phase 3 is complete, proceed to **GPU Phase 4: Integration & Optimization** which will integrate GPU training into the main training loop and optimize performance.
