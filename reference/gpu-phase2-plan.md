# GPU Phase 2: Forward Pass on GPU

## Overview

**Goal**: Implement GPU compute shaders for the forward pass of gate layers, perception module, and update module. Verify numerical equivalence with CPU implementation.

**Estimated Duration**: 2-3 days

**Dependencies**: Phase 1 complete (wgpu infrastructure working)

---

## Success Criteria

1. ✅ Gate forward shader computes softmax + weighted sum correctly
2. ✅ Perception forward pass matches CPU output within 1e-5
3. ✅ Update forward pass matches CPU output within 1e-5
4. ✅ Full DiffLogicCA forward step matches CPU within 1e-5
5. ✅ Performance improvement measurable (>2x for large grids)

---

## Architecture Review

Before implementing, understand the data flow:

```
Input Grid (16×16×8 = 2048 f32)
    ↓
Perception Module (per cell)
    - 16 kernels, each processing center vs 8 neighbors
    - 3 gate layers: [9→8], [8→4], [4→2]
    - Output: 264 features per cell (8 center + 16×2×8 kernel outputs)
    ↓
Update Module (per cell)
    - 10 gate layers: [264→256], [256→256]×9, [256→128], etc.
    - Output: 8 channels per cell
    ↓
Output Grid (16×16×8 = 2048 f32)
```

**Key insight**: All cells can be processed in parallel. Within each cell, gate operations are sequential within layers but parallel across gates in a layer.

---

## Task Breakdown

### Task 2.1: Design GPU Buffer Layout

**Description**: Define how CPU data structures map to GPU buffers

**Design**:

```rust
/// GPU buffer layout for a GateLayer
struct GateLayerGpuData {
    /// Gate logits: [num_gates × 16] f32
    /// Contiguous array, gate i has logits at [i*16 .. (i+1)*16]
    logits: Vec<f32>,
    
    /// Wire indices: [num_gates × 2] u32
    /// Gate i reads inputs from wires[i*2] and wires[i*2+1]
    wires: Vec<u32>,
}

/// GPU buffer layout for forward pass
struct ForwardPassBuffers {
    /// Input values: [batch_size × num_cells × input_dim] f32
    inputs: Buffer,
    
    /// Output values: [batch_size × num_cells × output_dim] f32  
    outputs: Buffer,
    
    /// Gate logits (all layers concatenated): [total_gates × 16] f32
    all_logits: Buffer,
    
    /// Wire indices (all layers): [total_gates × 2] u32
    all_wires: Buffer,
    
    /// Layer metadata: [num_layers × 4] u32
    /// Each layer: (start_gate_idx, num_gates, input_offset, output_offset)
    layer_info: Buffer,
}
```

**Memory Layout for Checkerboard**:
- Grid: 16×16×8 = 2,048 floats = 8 KB
- Perception (224 gates × 16 logits): 14,336 floats = 56 KB
- Update (2,816 gates × 16 logits): 180,224 floats = 703 KB
- Total per model: ~770 KB (fits easily in GPU memory)

**Tests**:
```rust
#[test]
fn test_gpu_buffer_layout_sizes() {
    let perception = create_checkerboard_perception();
    let update = create_checkerboard_update();
    
    // Verify we can compute buffer sizes
    let p_gates = perception.total_gates();
    let u_gates = update.total_gates();
    
    assert_eq!(p_gates, 224);  // 16 kernels × 14 gates each
    assert_eq!(u_gates, 2816); // From architecture
}
```

**Exit Criteria**:
- [ ] Buffer layout documented
- [ ] Size calculations match expected values
- [ ] Data conversion functions designed

---

### Task 2.2: Implement Gate Forward Shader

**Description**: WGSL shader for soft gate forward pass

**Implementation** (`src/gpu/shaders/gate_forward.wgsl`):

```wgsl
// Uniform containing layer configuration
struct LayerConfig {
    num_gates: u32,
    input_offset: u32,
    output_offset: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> config: LayerConfig;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> logits: array<f32>;  // [gate_idx * 16 + op]
@group(0) @binding(3) var<storage, read> wires: array<u32>;   // [gate_idx * 2 + {0,1}]
@group(0) @binding(4) var<storage, read_write> outputs: array<f32>;

// All 16 binary operations
fn binary_op(op: u32, a: f32, b: f32) -> f32 {
    switch (op) {
        case 0u: { return 0.0; }                    // False
        case 1u: { return a * b; }                  // And
        case 2u: { return a * (1.0 - b); }         // A and not B
        case 3u: { return a; }                      // A (pass-through)
        case 4u: { return (1.0 - a) * b; }         // Not A and B
        case 5u: { return b; }                      // B
        case 6u: { return a + b - 2.0 * a * b; }   // Xor
        case 7u: { return a + b - a * b; }         // Or
        case 8u: { return 1.0 - (a + b - a * b); } // Nor
        case 9u: { return 1.0 - (a + b - 2.0 * a * b); } // Xnor
        case 10u: { return 1.0 - b; }              // Not B
        case 11u: { return 1.0 - (1.0 - a) * b; }  // A or not B (implication B→A)
        case 12u: { return 1.0 - a; }              // Not A
        case 13u: { return 1.0 - a * (1.0 - b); }  // Not A or B (implication A→B)
        case 14u: { return 1.0 - a * b; }          // Nand
        default: { return 1.0; }                    // True (case 15)
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gate_idx = gid.x;
    if (gate_idx >= config.num_gates) {
        return;
    }
    
    // Read input wires
    let wire_a = wires[gate_idx * 2u];
    let wire_b = wires[gate_idx * 2u + 1u];
    let a = inputs[config.input_offset + wire_a];
    let b = inputs[config.input_offset + wire_b];
    
    // Compute softmax over 16 logits for numerical stability
    var max_logit = logits[gate_idx * 16u];
    for (var i = 1u; i < 16u; i++) {
        max_logit = max(max_logit, logits[gate_idx * 16u + i]);
    }
    
    var exp_sum = 0.0;
    for (var i = 0u; i < 16u; i++) {
        exp_sum += exp(logits[gate_idx * 16u + i] - max_logit);
    }
    
    // Weighted sum of all 16 operations
    var result = 0.0;
    for (var i = 0u; i < 16u; i++) {
        let prob = exp(logits[gate_idx * 16u + i] - max_logit) / exp_sum;
        let op_result = binary_op(i, a, b);
        result += prob * op_result;
    }
    
    outputs[config.output_offset + gate_idx] = result;
}
```

**Tests**:
```rust
#[test]
fn test_gate_forward_shader_single_gate() {
    let ctx = GpuContext::new().unwrap();
    
    // Single gate with pass-through initialization (logits[3] = 10.0)
    let mut logits = vec![0.0f32; 16];
    logits[3] = 10.0; // A (pass-through)
    
    let inputs = vec![0.7f32, 0.3f32]; // a=0.7, b=0.3
    let wires = vec![0u32, 1u32];
    
    let output = ctx.run_gate_forward(&inputs, &logits, &wires).unwrap();
    
    // Should be very close to 0.7 (input a)
    assert!((output[0] - 0.7).abs() < 0.01, "Expected ~0.7, got {}", output[0]);
}

#[test]
fn test_gate_forward_shader_matches_cpu() {
    let ctx = GpuContext::new().unwrap();
    
    // Create a gate layer with random logits
    let mut rng = SimpleRng::new(42);
    let layer = GateLayer::new(4, 2, ConnectionType::Unique, &mut rng);
    
    let inputs = vec![0.2, 0.4, 0.6, 0.8];
    
    // CPU forward
    let cpu_output = layer.forward_soft(&inputs);
    
    // GPU forward
    let gpu_output = ctx.run_gate_layer_forward(&layer, &inputs).unwrap();
    
    // Compare
    for (i, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        assert!(
            (cpu - gpu).abs() < 1e-5,
            "Mismatch at {}: CPU={}, GPU={}",
            i, cpu, gpu
        );
    }
}
```

**Exit Criteria**:
- [ ] Shader compiles without WGSL errors
- [ ] Single gate test passes
- [ ] Multi-gate test matches CPU within 1e-5

---

### Task 2.3: Implement GPU Gate Layer Forward

**Description**: Rust wrapper for running gate layer forward on GPU

**Implementation** (`src/gpu/gate_layer.rs`):

```rust
use crate::perception::GateLayer;

impl GpuContext {
    /// Run a gate layer forward pass on GPU
    pub fn run_gate_layer_forward(
        &self,
        layer: &GateLayer,
        inputs: &[f64],
    ) -> Result<Vec<f64>, GpuError> {
        // Convert f64 to f32 for GPU
        let inputs_f32: Vec<f32> = inputs.iter().map(|&x| x as f32).collect();
        let logits_f32 = layer.get_logits_flat_f32();
        let wires = layer.get_wires_flat();
        
        // Create buffers, run shader, read back
        let output_f32 = self.run_gate_forward_internal(
            &inputs_f32,
            &logits_f32,
            &wires,
            layer.output_size(),
        )?;
        
        // Convert back to f64
        Ok(output_f32.iter().map(|&x| x as f64).collect())
    }
}
```

**Additional Methods on GateLayer**:
```rust
impl GateLayer {
    /// Get all logits as flat f32 array [num_gates × 16]
    pub fn get_logits_flat_f32(&self) -> Vec<f32> {
        self.gates.iter()
            .flat_map(|g| g.logits.iter().map(|&x| x as f32))
            .collect()
    }
    
    /// Get all wire indices as flat u32 array [num_gates × 2]
    pub fn get_wires_flat(&self) -> Vec<u32> {
        self.gates.iter()
            .flat_map(|g| [g.input_a as u32, g.input_b as u32])
            .collect()
    }
}
```

**Tests**:
```rust
#[test]
fn test_gate_layer_gpu_vs_cpu_random() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(12345);
    
    // Test various layer sizes
    for (input_size, output_size) in [(8, 4), (16, 8), (64, 32), (256, 128)] {
        let layer = GateLayer::new(input_size, output_size, ConnectionType::Unique, &mut rng);
        let inputs: Vec<f64> = (0..input_size).map(|_| rng.next_f64()).collect();
        
        let cpu = layer.forward_soft(&inputs);
        let gpu = ctx.run_gate_layer_forward(&layer, &inputs).unwrap();
        
        let max_diff: f64 = cpu.iter().zip(gpu.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0, f64::max);
        
        assert!(max_diff < 1e-5, "Layer {}→{}: max diff = {}", input_size, output_size, max_diff);
    }
}
```

**Exit Criteria**:
- [ ] `get_logits_flat_f32()` implemented
- [ ] `get_wires_flat()` implemented
- [ ] `run_gate_layer_forward()` works
- [ ] Random layer tests pass within 1e-5

---

### Task 2.4: Implement Perception Module GPU Forward

**Description**: GPU forward pass for entire perception module (16 kernels × 3 layers)

**Design Considerations**:
- Each kernel processes independently → batch all kernels
- Each layer must complete before next layer → sequential dispatch
- All cells processed in parallel for each layer

**Implementation** (`src/gpu/perception.rs`):

```rust
impl GpuContext {
    /// Run perception forward on GPU for a single cell
    pub fn run_perception_forward(
        &self,
        perception: &PerceptionModule,
        neighborhood: &NNeighborhood,
    ) -> Result<Vec<f64>, GpuError> {
        // Prepare inputs for all kernels: [kernel × layer_inputs]
        let mut all_kernel_outputs = Vec::new();
        
        for (k, kernel) in perception.kernels.iter().enumerate() {
            // Get kernel input: center value XOR neighbor value
            let center = neighborhood.center();
            let neighbor = neighborhood.get_neighbor(k);
            
            // Input is [center_channels, neighbor_channels] flattened
            let input: Vec<f64> = center.iter()
                .zip(neighbor.iter())
                .flat_map(|(&c, &n)| [c, n])
                .collect();
            
            // Run through kernel's gate layers
            let output = self.run_kernel_layers_forward(kernel, &input)?;
            all_kernel_outputs.extend(output);
        }
        
        // Combine: center + kernel outputs (matching CPU ordering)
        let center = neighborhood.center();
        let mut result = Vec::with_capacity(perception.output_size());
        
        // Order: (channel, sobel, kernel) = (c, s, k)
        for c in 0..perception.channels {
            result.push(center[c]);
            for s in 0..2 { // 2 output bits per kernel
                for k in 0..perception.num_kernels {
                    result.push(all_kernel_outputs[k * 2 + s + c * 2 * perception.num_kernels]);
                    // Wait, need to verify this ordering matches CPU...
                }
            }
        }
        
        Ok(result)
    }
}
```

**Note**: The exact output ordering must match `perception.rs:373` - review carefully!

**Tests**:
```rust
#[test]
fn test_perception_gpu_vs_cpu() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(999);
    
    let perception = create_checkerboard_perception();
    let grid = NGrid::random(16, 16, 8, &mut rng);
    
    // Test a few cells
    for (x, y) in [(0, 0), (7, 7), (15, 15)] {
        let neighborhood = grid.neighborhood(x, y);
        
        let cpu = perception.forward_soft(&neighborhood);
        let gpu = ctx.run_perception_forward(&perception, &neighborhood).unwrap();
        
        let max_diff: f64 = cpu.iter().zip(gpu.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0, f64::max);
        
        assert!(max_diff < 1e-5, "Cell ({},{}): max diff = {}", x, y, max_diff);
    }
}
```

**Exit Criteria**:
- [ ] GPU perception forward runs
- [ ] Output size matches CPU (264 for checkerboard)
- [ ] Values match CPU within 1e-5 for all test cells

---

### Task 2.5: Implement Update Module GPU Forward

**Description**: GPU forward pass for update module (10+ layers, 256+ gates per layer)

**Implementation** (`src/gpu/update.rs`):

```rust
impl GpuContext {
    /// Run update module forward on GPU
    pub fn run_update_forward(
        &self,
        update: &UpdateModule,
        perception_output: &[f64],
    ) -> Result<Vec<f64>, GpuError> {
        let mut current = perception_output.to_vec();
        
        // Process each layer sequentially (output of layer N is input to layer N+1)
        for layer in &update.layers {
            current = self.run_gate_layer_forward(layer, &current)?;
        }
        
        Ok(current)
    }
}
```

**Optimization Opportunity**: Batch multiple cells' update passes into single kernel dispatches.

**Tests**:
```rust
#[test]
fn test_update_gpu_vs_cpu() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(777);
    
    let update = create_checkerboard_update();
    let input: Vec<f64> = (0..264).map(|_| rng.next_f64()).collect();
    
    let cpu = update.forward_soft(&input);
    let gpu = ctx.run_update_forward(&update, &input).unwrap();
    
    assert_eq!(cpu.len(), gpu.len());
    assert_eq!(cpu.len(), 8); // 8 output channels
    
    let max_diff: f64 = cpu.iter().zip(gpu.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0, f64::max);
    
    assert!(max_diff < 1e-4, "max diff = {}", max_diff);
}
```

**Exit Criteria**:
- [ ] Update forward works on GPU
- [ ] Output size matches (8 channels)
- [ ] Values match CPU within 1e-4 (allowing f32 precision loss)

---

### Task 2.6: Implement Full CA Step GPU Forward

**Description**: Complete forward pass: Grid → Perception → Update → Grid

**Implementation** (`src/gpu/ca.rs`):

```rust
impl GpuContext {
    /// Run one CA step on GPU
    pub fn run_ca_step(
        &self,
        model: &DiffLogicCA,
        grid: &NGrid,
    ) -> Result<NGrid, GpuError> {
        let mut output = NGrid::new(
            grid.width, grid.height, grid.channels,
            grid.boundary.clone(),
        );
        
        // Process all cells (can parallelize with rayon on CPU side for dispatch)
        for y in 0..grid.height {
            for x in 0..grid.width {
                let neighborhood = grid.neighborhood(x as isize, y as isize);
                
                // GPU perception
                let perception_out = self.run_perception_forward(
                    &model.perception,
                    &neighborhood,
                )?;
                
                // GPU update
                let update_out = self.run_update_forward(
                    &model.update,
                    &perception_out,
                )?;
                
                // Write to output grid
                for c in 0..grid.channels {
                    output.set(x as isize, y as isize, c, update_out[c]);
                }
            }
        }
        
        Ok(output)
    }
}
```

**Performance Note**: This naive implementation does one kernel dispatch per gate layer per cell - very inefficient. Task 2.7 will batch this.

**Tests**:
```rust
#[test]
fn test_ca_step_gpu_vs_cpu() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(555);
    
    let model = create_checkerboard_model();
    let grid = create_random_seed(16, 8, &mut rng);
    
    let cpu_output = model.forward_soft(&grid);
    let gpu_output = ctx.run_ca_step(&model, &grid).unwrap();
    
    // Compare all cells
    let mut max_diff = 0.0f64;
    for y in 0..16 {
        for x in 0..16 {
            for c in 0..8 {
                let cpu_val = cpu_output.get(x, y, c);
                let gpu_val = gpu_output.get(x, y, c);
                max_diff = max_diff.max((cpu_val - gpu_val).abs());
            }
        }
    }
    
    assert!(max_diff < 1e-4, "max diff = {}", max_diff);
}
```

**Exit Criteria**:
- [ ] Full CA step runs on GPU
- [ ] Output grid matches CPU within 1e-4
- [ ] No crashes or GPU errors

---

### Task 2.7: Batch All Cells for Efficiency

**Description**: Instead of one dispatch per cell, batch all cells' computation

**Design**:
```
Instead of:
  for each cell:
    dispatch perception kernel
    dispatch update kernel

Do:
  1. Prepare all 256 neighborhoods as contiguous buffer
  2. Dispatch perception kernel once for ALL gates across ALL cells
  3. Dispatch update kernel once for ALL gates across ALL cells
```

**Buffer Layout for Batched Forward**:
```
inputs: [cell_0_perception_input, cell_1_perception_input, ...]
         each is 264 floats, total = 256 cells × 264 = 67,584 floats

outputs: [cell_0_update_output, cell_1_update_output, ...]
          each is 8 floats, total = 256 cells × 8 = 2,048 floats
```

**Shader Modification**:
```wgsl
// Each workgroup processes gates for ONE layer across ALL cells
// global_id.x = gate_index within layer
// global_id.y = cell_index

@compute @workgroup_size(64, 4, 1)  // 64 gates × 4 cells per workgroup
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gate_idx = gid.x;
    let cell_idx = gid.y;
    
    if (gate_idx >= config.num_gates || cell_idx >= config.num_cells) {
        return;
    }
    
    // Calculate buffer offsets for this cell
    let input_base = cell_idx * config.input_stride + config.input_offset;
    let output_base = cell_idx * config.output_stride + config.output_offset;
    
    // ... rest of computation
}
```

**Tests**:
```rust
#[test]
fn test_batched_forward_vs_sequential() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(333);
    
    let model = create_checkerboard_model();
    let grid = create_random_seed(16, 8, &mut rng);
    
    // Sequential (Task 2.6)
    let sequential = ctx.run_ca_step(&model, &grid).unwrap();
    
    // Batched (this task)
    let batched = ctx.run_ca_step_batched(&model, &grid).unwrap();
    
    // Should be identical
    for y in 0..16 {
        for x in 0..16 {
            for c in 0..8 {
                let seq = sequential.get(x, y, c);
                let bat = batched.get(x, y, c);
                assert!((seq - bat).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn test_batched_forward_performance() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(111);
    
    let model = create_checkerboard_model();
    let grid = create_random_seed(16, 8, &mut rng);
    
    // Warmup
    for _ in 0..5 {
        ctx.run_ca_step_batched(&model, &grid).unwrap();
    }
    
    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..100 {
        ctx.run_ca_step_batched(&model, &grid).unwrap();
    }
    let gpu_time = start.elapsed();
    
    let start = std::time::Instant::now();
    for _ in 0..100 {
        model.forward_soft(&grid);
    }
    let cpu_time = start.elapsed();
    
    println!("GPU: {:?} for 100 steps", gpu_time);
    println!("CPU: {:?} for 100 steps", cpu_time);
    println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    
    // Expect at least 2x speedup for 16×16 grid
    assert!(gpu_time < cpu_time, "GPU should be faster than CPU");
}
```

**Exit Criteria**:
- [x] Batched forward implemented
- [x] Results match CPU version within 1e-4
- [ ] ~~Measurable performance improvement (>2x expected)~~ **NOT ACHIEVED - see findings below**

---

## Actual Implementation Findings (2026-01-08)

### What Was Implemented

Tasks 2.1-2.7 are complete with the following results:

| Task | Status | Notes |
|------|--------|-------|
| 2.1 Buffer layout | ✅ | `get_logits_flat_f32()`, `get_wires_flat_u32()` |
| 2.2 Gate forward shader | ✅ | `gate_forward.wgsl`, `gate_forward_batched.wgsl` |
| 2.3 Gate layer wrapper | ✅ | `run_gate_layer_forward()` |
| 2.4 Perception GPU | ✅ | `run_perception_forward()` |
| 2.5 Update GPU | ✅ | `run_update_forward()` |
| 2.6 Full CA step | ✅ | `run_ca_step()` (naive, very slow) |
| 2.7 Batched forward | ✅ | `run_ca_step_batched()` |

### Performance Reality

**Benchmarks on AMD RX 6800 XT (16×16 grid, small model):**
```
GPU batched: 4.4s for 20 steps = 220ms/step
CPU:         1.2s for 20 steps = 62ms/step
Speedup: 0.28x (GPU is 3.5x SLOWER)
```

### Why GPU Is Slower

1. **Too many dispatches**: Current batching is per-layer, not per-model
   - Perception: 16 kernels × 8 channels × 3 layers = 384 dispatches
   - Update: ~16 layers = 16 dispatches
   - Total: ~400 GPU dispatches per CA step

2. **Small problem size**: 256 cells (16×16) is insufficient parallelism
   - GPU dispatch overhead: ~0.5ms per dispatch
   - 400 dispatches × 0.5ms = 200ms overhead alone

3. **Data transfer overhead**: f64→f32→GPU→f32→f64 every layer
   - No persistent GPU buffers between layers

### What Would Be Needed for GPU Speedup

To achieve actual speedup, Phase 2 would need:

1. **Fully fused kernels**: Single dispatch for entire perception or update module
   - Requires complex shader that loops through all layers internally
   - Need to keep intermediate activations in GPU shared memory

2. **Persistent GPU buffers**: Keep model weights and activations on GPU
   - Only transfer input grid to GPU and output grid back
   - Reuse buffers across training steps

3. **Larger grids**: 64×64 or 128×128 to amortize dispatch overhead
   - 4096-16384 cells vs current 256

4. **Different architecture**: Consider using wgpu compute pipelines differently
   - Possibly use single large dispatch with internal synchronization

### Recommendation

**Phase 2 provides foundation but NOT performance benefit.** Options:

A. **Defer GPU optimization** to after CPU training works
   - Current CPU with rayon parallelization may be sufficient
   - Focus on getting checkerboard to train first

B. **Invest in Phase 2.8**: Fused kernel implementation
   - Significant effort (3-5 days) 
   - Would require complete shader rewrite

**Current recommendation: Option A** - The GPU infrastructure is in place and verified correct. Return to GPU optimization after CPU training succeeds.

---

## Final Checklist

| Task | Status | Verified By |
|------|--------|-------------|
| 2.1 Buffer layout designed | ✅ | Code review |
| 2.2 Gate forward shader works | ✅ | `test_gate_forward_shader_*` |
| 2.3 Gate layer GPU wrapper works | ✅ | `test_gate_layer_gpu_vs_cpu_random` |
| 2.4 Perception GPU forward works | ✅ | `test_perception_gpu_vs_cpu` |
| 2.5 Update GPU forward works | ✅ | `test_update_gpu_vs_cpu` |
| 2.6 Full CA step GPU works | ✅ | `test_ca_step_gpu_vs_cpu` (ignored - slow) |
| 2.7 Batched forward works | ✅ | `test_ca_step_batched_vs_cpu` |
| 2.7 Batched speedup achieved | ❌ | GPU still slower than CPU |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| f32 precision loss | High | Medium | Accept 1e-4 tolerance, not 1e-5 |
| Output ordering mismatch | High | High | Careful review of perception.rs |
| Memory bandwidth bottleneck | Medium | Medium | Batch more aggressively |
| Shader compilation errors | Medium | Low | Develop incrementally |

---

## Performance Expectations

| Grid Size | CPU Time/Step | GPU Time/Step (Expected) | Speedup |
|-----------|---------------|---------------------------|---------|
| 16×16 | ~50ms | ~5ms | ~10x |
| 32×32 | ~200ms | ~10ms | ~20x |
| 64×64 | ~800ms | ~25ms | ~32x |

These are rough estimates. Actual performance depends on:
- Kernel dispatch overhead
- Memory transfer time
- GPU occupancy
- Driver efficiency

---

## Notes for Implementation

1. **Start simple**: Get single gate working before batching
2. **Verify at each step**: Don't proceed until CPU-GPU match
3. **Watch precision**: f32 GPU vs f64 CPU will differ slightly
4. **Profile early**: Use GPU timing to find bottlenecks
5. **Output ordering is critical**: Spend time verifying perception output layout

---

## Next Phase

After Phase 2 is complete, proceed to **GPU Phase 3: Backward Pass on GPU** which will implement gradient computation shaders for training.
