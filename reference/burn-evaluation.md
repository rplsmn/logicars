# Burn Framework Evaluation for Logicars

**Date**: 2026-01-08
**Context**: After completing wgpu GPU Phase 2 (forward pass), evaluating alternative approaches.

## Summary

**Recommendation**: Defer Burn adoption until CPU training converges. The current blocker is training dynamics (gradient cancellation), not performance.

---

## Current State

### wgpu Implementation (Phase 2)

- ✅ Forward pass implemented and numerically correct
- ✅ Batched processing for all cells in parallel
- ❌ **GPU is 3.5x slower than CPU** for 16×16 grids

**Root cause of slowness**:
1. ~400 GPU dispatches per CA step (one per layer)
2. Small problem size (256 cells insufficient parallelism)
3. Data transfer overhead every layer

**What Phase 3-4 would require**:
- Custom backward pass WGSL shaders
- Fused kernels (rewrite entire pipeline)
- Persistent GPU buffers
- ~3-5 additional days of complex work

### Training Status

CPU training is not converging due to:
- Gradient cancellation (50% black/white targets)
- Pass-through gates stuck at logit=10.0
- Expected ~200 epoch plateau before escape

**GPU won't fix training dynamics issues.**

---

## Burn Framework Overview

[Burn](https://burn.dev/) is a Rust deep learning framework with:

### Core Features

1. **Multiple Backends**
   - `Wgpu` - Cross-platform GPU (WebGPU/Vulkan/Metal)
   - `Cuda` - NVIDIA native
   - `Rocm` - AMD native (perfect for RX 7800 XT)
   - `NdArray` - CPU fallback

2. **Automatic Differentiation**
   ```rust
   type Backend = Autodiff<Wgpu>;
   let y = model.forward(x);
   let grads = y.backward();  // Automatic!
   ```

3. **Kernel Fusion**
   ```rust
   type Backend = Fusion<Wgpu>;
   // Operations automatically fused into single GPU dispatch
   ```

4. **Built-in Training**
   - AdamW, SGD, other optimizers
   - Training TUI with progress
   - Checkpointing

### How It Would Solve Our Problems

| Problem | Current wgpu | Burn Solution |
|---------|--------------|---------------|
| 400 dispatches | Manual fused shaders | `Fusion<B>` auto-fuses |
| Manual gradients | WGSL backward shaders | `Autodiff<B>` automatic |
| AMD GPU support | wgpu only | Native ROCm backend |
| f64→f32 conversion | Every transfer | Native f32 tensors |

---

## Integration Analysis

### What Would Change

1. **Core Types**
   ```rust
   // Current
   struct NGrid { data: Vec<f64>, ... }
   
   // Burn
   type Grid<B> = Tensor<B, 3>;  // [H, W, C]
   ```

2. **Gate Layer**
   ```rust
   // Custom Burn module for 16-op softmax gate
   #[derive(Module)]
   struct DiffLogicGate<B: Backend> {
       logits: Param<Tensor<B, 2>>,  // [num_gates, 16]
       wires: Tensor<B, 2, Int>,     // [num_gates, 2]
   }
   
   impl<B: Backend> DiffLogicGate<B> {
       fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
           // Softmax over logits
           let probs = self.logits.val().softmax(1);
           // Compute all 16 ops and weighted sum
           // ... (custom implementation)
       }
   }
   ```

3. **Training Loop**
   ```rust
   // Burn handles gradients automatically
   let loss = model.forward(input).mse_loss(target);
   let grads = loss.backward();
   optimizer.step(grads);
   ```

### Effort Estimate

| Task | Days | Notes |
|------|------|-------|
| Learn Burn API | 1-2 | Good docs, examples |
| Core types refactor | 2-3 | NGrid → Tensor |
| Custom gate module | 1-2 | 16-op softmax |
| Perception/Update modules | 1-2 | As Burn Modules |
| Training integration | 1-2 | BPTT with autodiff |
| Testing/debugging | 2-3 | Verify numerical equivalence |
| **Total** | **~10 days** | |

---

## Trade-offs

### Advantages of Burn

1. **No manual gradient code** - Autodiff handles everything
2. **Automatic kernel fusion** - Single dispatch for fused operations
3. **Multi-backend** - Easy switch between wgpu/CUDA/ROCm
4. **Production-ready** - Built-in training, checkpointing, TUI
5. **Active development** - Well-maintained, good community

### Disadvantages

1. **Refactoring required** - Core types need rewrite
2. **Custom ops needed** - 16-op gate is not standard
3. **Learning curve** - New API to learn
4. **Dependency** - Adds significant dependency

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Custom gate ops hard to express | Medium | Burn supports custom modules |
| Performance regression | Low | Burn is optimized for GPU |
| API changes | Low | Burn is mature (v0.15+) |
| Algorithm still doesn't converge | High | Verify on CPU first |

---

## Recommendation

### Phase 1: CPU Training First (Current Priority)

1. Run 500+ epoch training on CPU
2. Verify algorithm converges (expect escape at ~200 epochs)
3. Once working, consider GPU optimization

### Phase 2: Burn Integration (Future)

When CPU training works:

1. **Option A**: Full Burn port (~10 days)
   - Complete rewrite with Burn tensors
   - Automatic GPU acceleration
   - Best long-term solution

2. **Option B**: Hybrid approach (~5 days)
   - Keep CPU implementation
   - Use Burn only for training acceleration
   - Lower refactoring effort

### Archive Current wgpu Work

The current `src/gpu/` implementation:
- ✅ Proves concept works
- ✅ Verified numerically correct
- ❌ Not worth further investment
- → Keep as reference but don't continue Phase 3-4

---

## Quick Start (Future Reference)

When ready to integrate Burn:

```toml
# Cargo.toml
[dependencies]
burn = { version = "0.15", features = ["wgpu", "autodiff", "fusion", "train"] }
```

```rust
// Example training step
use burn::backend::{Autodiff, wgpu::Wgpu};
use burn::optim::AdamWConfig;

type Backend = Autodiff<Wgpu>;

fn train_step(model: &DiffLogicCA<Backend>, input: Tensor<Backend, 3>, target: Tensor<Backend, 3>) {
    let output = model.forward(input);
    let loss = (output - target).powf_scalar(2.0).sum();
    
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    
    let mut optim = AdamWConfig::new().init();
    model.update(&grads, &mut optim);
}
```

---

## Conclusion

Burn is a compelling solution for GPU acceleration, offering automatic differentiation and kernel fusion that directly address our current wgpu implementation's limitations. However, the immediate priority should be getting CPU training to converge - GPU speedup is pointless if the algorithm doesn't work.

**Action items**:
1. Archive current GPU work (don't continue Phase 3-4)
2. Focus on CPU training convergence (run 500+ epochs)
3. Revisit Burn integration when training works
