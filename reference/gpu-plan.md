# GPU Acceleration Planning for Logicars

## Summary

This document analyzes the differences between our Rust CPU implementation and the JAX GPU reference, and plans a GPU acceleration path using AMD Radeon RX 7800 XT.

---

## Part 1: JAX vs Rust CPU - Why Results May Differ

### 1.1 JAX's XLA Backend and GPU Execution Model

JAX uses XLA (Accelerated Linear Algebra) to compile Python/NumPy operations into optimized GPU kernels. Key differences from our sequential Rust implementation:

#### **1.1.1 Parallel Execution of All Operations**

In JAX:
```python
# All 16 gate operations computed simultaneously as a batched GEMM
combinations = bin_op_all_combinations(a, b)  # GPU: Single kernel
result = (combinations * probs).sum(axis=-1)   # GPU: Fused reduction
```

On GPU, ALL gates across ALL cells are computed in a single kernel launch. The parallelism is not just at the cell level but at the operation level within cells.

Our Rust implementation processes cells sequentially (now parallel with rayon), but within each cell, gate operations are still sequential.

#### **1.1.2 Numerical Precision and Ordering**

GPU floating-point operations have different accumulation ordering than CPU:

1. **GPU uses fused multiply-add (FMA)**: `a*b + c` is computed in one operation with only one rounding, vs two separate operations on CPU.

2. **Reduction trees**: GPU reductions use tree-based parallel sums, which have different numerical properties than sequential sums:
   ```
   CPU:  ((((a + b) + c) + d) + e)  # Left-to-right
   GPU:  ((a + b) + (c + d)) + e    # Tree reduction
   ```

3. **Order of gradient accumulation**: In backpropagation, gradients from different cells are accumulated. GPU accumulates in parallel with different ordering.

#### **1.1.3 Autodiff Implementation Differences**

JAX uses source-to-source automatic differentiation with specific optimizations:

1. **Gradient graph optimizations**: JAX's XLA compiler fuses gradient operations, potentially changing numerical behavior.

2. **Memory layout**: JAX stores tensors in GPU-optimized layouts (potentially column-major or tiled), affecting cache behavior and numerical ordering.

3. **Batch normalization of gradients**: The way gradients are accumulated across the batch affects the final update.

### 1.2 Specific Impact on Checkerboard Training

#### **1.2.1 Gradient Cancellation Problem**

With checkerboard (50% black, 50% white) and outputs at 0.5:
- Cells with target=0 → positive gradient
- Cells with target=1 → negative gradient

On CPU, these are accumulated sequentially and may cancel exactly. On GPU, tree reduction produces different intermediate sums, potentially leaving residual gradients that help escape local minima.

#### **1.2.2 Pass-Through Gate Escape**

The high initial logit (10.0) for pass-through gates creates softmax saturation:
```
prob[3] = exp(10) / (exp(10) + 15*exp(0)) ≈ 0.9993
```

Weight decay reduces this by ~0.5% per epoch. But gradient-based updates depend on residual gradients from the cancellation problem above. GPU's different accumulation may produce larger residual gradients.

### 1.3 Why Reference Shows Loss Collapse at ~200 Epochs

Looking at `checkerboard_loss.png`:
1. **Epochs 0-200**: Hard loss flat, soft loss decreasing - gates remain pass-through
2. **Epoch ~200**: Sudden collapse - gates escape pass-through, start computing useful operations
3. **Epochs 200-500**: Both losses converge together

The escape at epoch ~200 is when weight decay has reduced the pass-through logit enough that:
1. Softmax is less saturated (gradients can flow)
2. Other gate logits have accumulated enough gradient signal

Our Rust implementation may have slower logit gap closure due to exact gradient cancellation.

---

## Part 2: AMD RX 7800 XT GPU Options

### 2.1 Hardware Specifications

| Spec | RX 7800 XT |
|------|------------|
| Architecture | RDNA 3 |
| Compute Units | 60 |
| Stream Processors | 3840 |
| Memory | 16GB GDDR6 |
| Memory Bandwidth | 624 GB/s |
| FP32 TFLOPS | 37.3 |
| ROCm Support | Yes (6.0+) |

### 2.2 GPU Framework Options for Rust

#### **Option A: wgpu (WebGPU)**

**Pros:**
- Cross-platform (works on AMD, NVIDIA, Intel, Metal)
- Rust-native with excellent ergonomics
- No vendor lock-in
- Works on Windows, Linux, macOS

**Cons:**
- Compute shader programming (WGSL)
- Less mature than CUDA ecosystem
- Manual gradient computation required

**Effort**: Medium-High

#### **Option B: ROCm + HIP**

**Pros:**
- Direct AMD support
- Similar to CUDA (HIP is CUDA-like API)
- Optimized for AMD hardware

**Cons:**
- AMD-only
- Linux-only (ROCm doesn't support Windows well)
- Requires C++ interop via bindgen

**Effort**: High

#### **Option C: OpenCL via ocl-rs**

**Pros:**
- Vendor-neutral
- Mature ecosystem
- Works on AMD, Intel, NVIDIA

**Cons:**
- Lower performance than native solutions
- Verbose API
- Less active development

**Effort**: Medium

#### **Option D: Vulkan Compute**

**Pros:**
- Low-level control
- Works on all modern GPUs
- Good AMD support

**Cons:**
- Very verbose
- Manual memory management
- No autodiff support

**Effort**: Very High

### 2.3 Recommended Approach: wgpu

For this project, **wgpu** is the best choice because:

1. **Rust-native**: No FFI complexity
2. **Cross-platform**: Works on your AMD GPU and can run on other hardware
3. **Active development**: Part of the Rust gamedev ecosystem
4. **Good compute shader support**: WGSL is expressive enough for our needs

---

## Part 3: GPU Implementation Plan

### 3.1 What to Move to GPU

Priority order based on compute intensity:

1. **Gate layer forward pass**: 16 operations × num_gates × num_cells
2. **Gate layer backward pass**: Gradient computation per gate
3. **Grid operations**: Neighborhood extraction, output assembly
4. **Loss computation**: Already parallel, but GPU would be faster

### 3.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Host (CPU)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Training    │  │ Optimizer   │  │ Data Loading    │ │
│  │ Loop        │  │ (AdamW)     │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│         │               │                               │
│         ▼               ▼                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │                 GPU Interface                     │  │
│  │  - Buffer management                             │  │
│  │  - Kernel dispatch                               │  │
│  │  - Synchronization                               │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    GPU (RX 7800 XT)                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Compute Shaders:                                │   │
│  │  - gate_forward.wgsl: Soft gate evaluation      │   │
│  │  - gate_backward.wgsl: Gradient computation     │   │
│  │  - perception.wgsl: Multi-kernel perception     │   │
│  │  - update.wgsl: Update network forward/backward │   │
│  │  - loss.wgsl: Parallel loss reduction           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Memory Buffers:                                       │
│  - Grid state (width × height × channels × f32)        │
│  - Gate logits (num_gates × 16 × f32)                 │
│  - Gate gradients (num_gates × 16 × f32)              │
│  - Wire indices (num_gates × 2 × u32)                 │
│  - Activations (for backprop)                         │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Key GPU Kernels

#### **3.3.1 Gate Forward Kernel**

```wgsl
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> logits: array<f32>;  // [gate * 16]
@group(0) @binding(2) var<storage, read> wires: array<u32>;   // [gate * 2]
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gate_idx = gid.x;
    if (gate_idx >= num_gates) { return; }
    
    let a = inputs[wires[gate_idx * 2]];
    let b = inputs[wires[gate_idx * 2 + 1]];
    
    // Compute softmax over 16 logits for this gate
    var max_logit = logits[gate_idx * 16];
    for (var i = 1u; i < 16u; i++) {
        max_logit = max(max_logit, logits[gate_idx * 16 + i]);
    }
    
    var exp_sum = 0.0;
    for (var i = 0u; i < 16u; i++) {
        exp_sum += exp(logits[gate_idx * 16 + i] - max_logit);
    }
    
    // Compute all 16 operations and weighted sum
    var result = 0.0;
    for (var i = 0u; i < 16u; i++) {
        let prob = exp(logits[gate_idx * 16 + i] - max_logit) / exp_sum;
        let op_result = compute_op(i, a, b);
        result += prob * op_result;
    }
    
    outputs[gate_idx] = result;
}
```

#### **3.3.2 Gate Backward Kernel**

Similar structure but computing `dL/dlogits` and `dL/dinputs`.

### 3.4 Implementation Phases

#### **Phase 1: Basic wgpu Setup** (1-2 days)
- Add wgpu dependency
- Create GPU context and device
- Implement simple test kernel
- Verify AMD GPU works

#### **Phase 2: Forward Pass on GPU** (2-3 days)
- Implement gate_forward.wgsl
- Port perception module
- Port update module
- Verify numerical equivalence with CPU

#### **Phase 3: Backward Pass on GPU** (3-4 days)
- Implement gradient computation kernels
- Port BPTT loop
- Handle gradient accumulation
- Verify gradient correctness

#### **Phase 4: Integration & Optimization** (2-3 days)
- Integrate with training loop
- Optimize memory transfers
- Batch multiple steps on GPU
- Profile and tune workgroup sizes

---

## Part 4: Will GPU Solve the Checkerboard Problem?

### 4.1 Likely: Yes, for These Reasons

1. **Different numerical accumulation**: GPU's tree-based reductions may not cancel gradients as exactly, leaving residual signal.

2. **More epochs per hour**: Even if the dynamics are the same, we can run 10-50x more epochs in the same wall-clock time, reaching the ~500 epoch convergence point faster.

3. **Batching is natural**: GPU can process entire batches in one kernel, making batch_size=20 or higher practical.

### 4.2 Alternative Fixes (If GPU Doesn't Help)

1. **Add gradient noise**: Perturb gradients slightly to break symmetry
   ```rust
   grad[i] += rng.next_f64() * 0.001 * grad[i].abs();
   ```

2. **Lower initial pass-through logit**: Use 5.0 instead of 10.0 (but this may hurt training stability)

3. **Curriculum learning**: Start with fewer steps (5), increase gradually

4. **Different loss function**: Focal loss or cross-entropy instead of MSE

5. **Learning rate warmup**: Start with higher LR to escape faster

---

## Part 5: Next Steps

1. **Create PR for rayon parallelization** - immediate speedup
2. **Run 1000 epoch training** - see if CPU eventually converges
3. **Start wgpu integration** - proof-of-concept kernel
4. **Benchmark GPU vs CPU** - measure actual speedup
5. **Full GPU training** - verify checkerboard convergence

---

## Appendix A: wgpu Cargo Dependencies

```toml
[dependencies]
wgpu = "24.0"
pollster = "0.4"  # For blocking on async GPU operations
bytemuck = { version = "1.18", features = ["derive"] }  # For GPU buffer casting
```

## Appendix B: Useful Resources

- [wgpu documentation](https://docs.rs/wgpu)
- [Learn wgpu](https://sotrh.github.io/learn-wgpu/)
- [WGSL specification](https://www.w3.org/TR/WGSL/)
- [ROCm for AMD](https://rocm.docs.amd.com/)
- [JAX XLA internals](https://jax.readthedocs.io/en/latest/jaxpr.html)
