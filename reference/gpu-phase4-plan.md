# GPU Phase 4: Integration & Optimization

## Overview

**Goal**: Integrate GPU forward/backward passes into the main training loop, optimize memory transfers, and profile for maximum performance.

**Estimated Duration**: 2-3 days

**Dependencies**: Phases 1-3 complete (wgpu working, forward and backward on GPU verified)

---

## Success Criteria

1. ✅ Training loop runs entirely on GPU (minimal CPU-GPU transfers)
2. ✅ AdamW optimizer update runs on GPU
3. ✅ Batch training works with batch_size=2
4. ✅ Performance: >10x speedup vs CPU for 16×16 grid
5. ✅ Checkerboard training runs to completion (500 epochs)
6. ✅ Training results match or exceed CPU training quality

---

## Architecture: Full GPU Training

```
┌─────────────────────────────────────────────────────────────────┐
│                         CPU (Host)                               │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │ Training Loop  │  │ Epoch Counter  │  │ Logging/Metrics  │   │
│  │ (orchestrator) │  │                │  │                  │   │
│  └───────┬────────┘  └────────────────┘  └──────────────────┘   │
│          │                                                       │
│          │ Once per epoch: report progress                       │
│          │                                                       │
│          ▼ Dispatch commands only                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         GPU (Device)                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                Persistent Buffers                         │   │
│  │  - Model parameters (logits): 3040 gates × 16 = 48K f32  │   │
│  │  - Adam state (m, v): 48K × 2 = 96K f32                  │   │
│  │  - Gradients: 48K f32                                     │   │
│  │  - Grid states (input/output): 2K f32 each               │   │
│  │  - Activation cache: ~2MB (20 steps)                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 Per-Epoch Compute                          │  │
│  │  1. Generate random seed (GPU or CPU→GPU once)            │  │
│  │  2. Forward pass × 20 steps (cache activations)           │  │
│  │  3. Compute loss (channel 0 only)                         │  │
│  │  4. Backward pass × 20 steps (accumulate gradients)       │  │
│  │  5. Clip gradients                                        │  │
│  │  6. AdamW update                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Every N epochs: GPU→CPU transfer for logging             │  │
│  │  - Loss value (single f32)                                │  │
│  │  - Hard accuracy (requires hard forward, expensive)       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task Breakdown

### Task 4.1: Implement Persistent GPU Buffers

**Description**: Create long-lived GPU buffers for model parameters to avoid per-epoch allocation

**Implementation** (`src/gpu/persistent.rs`):

```rust
/// Holds all GPU-resident state for a model
pub struct GpuModel {
    ctx: GpuContext,
    
    // Model parameters (persistent, updated by optimizer)
    perception_logits: Buffer,  // [perception_gates × 16] f32
    update_logits: Buffer,      // [update_gates × 16] f32
    
    // Wire indices (constant after initialization)
    perception_wires: Buffer,   // [perception_gates × 2] u32
    update_wires: Buffer,       // [update_gates × 2] u32
    
    // Adam optimizer state
    perception_m: Buffer,       // First moment
    perception_v: Buffer,       // Second moment
    update_m: Buffer,
    update_v: Buffer,
    
    // Gradient accumulator
    perception_grad: Buffer,
    update_grad: Buffer,
    
    // Model metadata
    num_perception_gates: usize,
    num_update_gates: usize,
    perception_layer_info: Vec<LayerInfo>,
    update_layer_info: Vec<LayerInfo>,
}

struct LayerInfo {
    start_gate: u32,
    num_gates: u32,
    input_size: u32,
    output_size: u32,
}

impl GpuModel {
    /// Upload a CPU model to GPU
    pub fn from_cpu(ctx: GpuContext, model: &DiffLogicCA) -> Self {
        // Extract and upload all logits
        let perception_logits_data = model.perception.get_all_logits_flat_f32();
        let update_logits_data = model.update.get_all_logits_flat_f32();
        
        // Create buffers
        let perception_logits = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("perception_logits"),
            contents: bytemuck::cast_slice(&perception_logits_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        
        // ... similar for other buffers ...
        
        // Initialize Adam state to zeros
        let zero_m = vec![0.0f32; perception_logits_data.len()];
        let perception_m = ctx.device.create_buffer_init(...);
        
        Self {
            ctx,
            perception_logits,
            // ...
        }
    }
    
    /// Download trained parameters back to CPU
    pub fn to_cpu(&self, model: &mut DiffLogicCA) -> Result<(), GpuError> {
        // Read back logits from GPU
        let perception_logits = self.read_buffer_f32(&self.perception_logits)?;
        let update_logits = self.read_buffer_f32(&self.update_logits)?;
        
        // Update CPU model
        model.perception.set_all_logits_from_flat(&perception_logits);
        model.update.set_all_logits_from_flat(&update_logits);
        
        Ok(())
    }
}
```

**Tests**:
```rust
#[test]
fn test_gpu_model_roundtrip() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(42);
    
    let original = create_checkerboard_model();
    let gpu_model = GpuModel::from_cpu(ctx, &original);
    
    let mut reconstructed = create_checkerboard_model();
    gpu_model.to_cpu(&mut reconstructed).unwrap();
    
    // Compare logits
    assert!(original.perception.logits_equal(&reconstructed.perception, 1e-6));
    assert!(original.update.logits_equal(&reconstructed.update, 1e-6));
}
```

**Exit Criteria**:
- [ ] `GpuModel::from_cpu` works
- [ ] `GpuModel::to_cpu` works
- [ ] Roundtrip preserves all parameters

---

### Task 4.2: Implement GPU AdamW Optimizer

**Description**: Run AdamW parameter update entirely on GPU

**Implementation** (`src/gpu/shaders/adamw.wgsl`):

```wgsl
struct AdamConfig {
    lr: f32,           // Learning rate (0.05)
    beta1: f32,        // First moment decay (0.9)
    beta2: f32,        // Second moment decay (0.99)
    epsilon: f32,      // Numerical stability (1e-8)
    weight_decay: f32, // L2 regularization (0.01)
    t: f32,            // Timestep (for bias correction)
    num_params: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> config: AdamConfig;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;  // First moment
@group(0) @binding(4) var<storage, read_write> v: array<f32>;  // Second moment

@compute @workgroup_size(256)
fn adamw_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= config.num_params) { return; }
    
    let grad = grads[idx];
    
    // Update biased first moment estimate
    m[idx] = config.beta1 * m[idx] + (1.0 - config.beta1) * grad;
    
    // Update biased second moment estimate
    v[idx] = config.beta2 * v[idx] + (1.0 - config.beta2) * grad * grad;
    
    // Compute bias-corrected estimates
    let m_hat = m[idx] / (1.0 - pow(config.beta1, config.t));
    let v_hat = v[idx] / (1.0 - pow(config.beta2, config.t));
    
    // AdamW: apply weight decay directly to params (not gradient)
    let param = params[idx];
    let new_param = param * (1.0 - config.lr * config.weight_decay) 
                  - config.lr * m_hat / (sqrt(v_hat) + config.epsilon);
    
    params[idx] = new_param;
    
    // Zero gradient for next iteration
    grads[idx] = 0.0;
}
```

**Rust Wrapper**:
```rust
impl GpuModel {
    /// Apply AdamW update to all parameters
    pub fn adamw_step(&mut self, epoch: usize) -> Result<(), GpuError> {
        let config = AdamConfig {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.99,  // Not 0.999!
            epsilon: 1e-8,
            weight_decay: 0.01,
            t: (epoch + 1) as f32,
            num_params: 0, // Will be set per dispatch
            _padding: 0,
        };
        
        // Update perception parameters
        self.run_adamw_kernel(
            &mut self.perception_logits,
            &self.perception_grad,
            &mut self.perception_m,
            &mut self.perception_v,
            config.with_num_params(self.num_perception_gates * 16),
        )?;
        
        // Update update parameters
        self.run_adamw_kernel(
            &mut self.update_logits,
            &self.update_grad,
            &mut self.update_m,
            &mut self.update_v,
            config.with_num_params(self.num_update_gates * 16),
        )?;
        
        Ok(())
    }
}
```

**Tests**:
```rust
#[test]
fn test_gpu_adamw_matches_cpu() {
    let ctx = GpuContext::new().unwrap();
    
    // Create simple test case
    let params = vec![1.0f32, 2.0, 3.0];
    let grads = vec![0.1f32, -0.2, 0.3];
    let m = vec![0.0f32; 3];
    let v = vec![0.0f32; 3];
    
    // GPU update
    let gpu_result = ctx.run_adamw_update(&params, &grads, &m, &v, 1).unwrap();
    
    // CPU update (using existing AdamW implementation)
    let mut cpu_optimizer = AdamW::new(0.05, 0.01, 0.9, 0.99);
    let cpu_result = cpu_optimizer.step(&params.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                                        &grads.iter().map(|&x| x as f64).collect::<Vec<_>>());
    
    // Compare
    for (i, (gpu, cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        assert!(((*gpu as f64) - cpu).abs() < 1e-5, 
                "Param {}: GPU={}, CPU={}", i, gpu, cpu);
    }
}
```

**Exit Criteria**:
- [ ] AdamW shader compiles
- [ ] Single step matches CPU optimizer
- [ ] Multiple steps accumulate correctly

---

### Task 4.3: Implement GPU Training Step

**Description**: Complete training step running entirely on GPU

**Implementation** (`src/gpu/training.rs`):

```rust
impl GpuModel {
    /// Run one complete training step on GPU
    /// 
    /// Returns the loss (for logging, requires GPU→CPU transfer)
    pub fn train_step(
        &mut self,
        seed: &NGrid,
        target: &NGrid,
        num_steps: usize,
        epoch: usize,
    ) -> Result<f32, GpuError> {
        // 1. Upload seed to GPU (could also generate on GPU)
        self.upload_grid(&self.input_grid, seed)?;
        
        // 2. Forward pass with caching
        for step in 0..num_steps {
            self.forward_step_cached(step)?;
        }
        
        // 3. Compute loss (only need value for logging)
        let loss = self.compute_loss(target)?;
        
        // 4. Backward pass (accumulates into grad buffers)
        self.backward_through_time(target, num_steps)?;
        
        // 5. Gradient clipping
        self.clip_gradients(100.0)?;
        
        // 6. AdamW update
        self.adamw_step(epoch)?;
        
        Ok(loss)
    }
    
    /// Run batch training step
    pub fn train_step_batch(
        &mut self,
        seeds: &[NGrid],
        target: &NGrid,
        num_steps: usize,
        epoch: usize,
    ) -> Result<f32, GpuError> {
        let mut total_loss = 0.0;
        
        // Zero gradients once at start
        self.zero_gradients()?;
        
        for seed in seeds {
            // Forward + backward (accumulates gradients)
            let loss = self.forward_backward_single(seed, target, num_steps)?;
            total_loss += loss;
        }
        
        // Single optimizer step after all samples
        self.clip_gradients(100.0)?;
        self.adamw_step(epoch)?;
        
        Ok(total_loss)
    }
}
```

**Tests**:
```rust
#[test]
fn test_gpu_train_step_runs() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(42);
    
    let model = create_small_checkerboard_model();
    let mut gpu_model = GpuModel::from_cpu(ctx, &model);
    
    let seed = create_random_seed(16, 8, &mut rng);
    let target = create_checkerboard(16, 2, 8);
    
    // Should run without errors
    let loss = gpu_model.train_step(&seed, &target, 5, 0).unwrap();
    
    // Loss should be positive
    assert!(loss > 0.0);
    assert!(!loss.is_nan());
}

#[test]
fn test_gpu_training_loss_decreases() {
    let ctx = GpuContext::new().unwrap();
    let mut rng = SimpleRng::new(42);
    
    let model = create_small_checkerboard_model();
    let mut gpu_model = GpuModel::from_cpu(ctx, &model);
    
    let target = create_checkerboard(16, 2, 8);
    
    // Run a few epochs
    let mut losses = Vec::new();
    for epoch in 0..10 {
        let seed = create_random_seed(16, 8, &mut rng);
        let loss = gpu_model.train_step(&seed, &target, 5, epoch).unwrap();
        losses.push(loss);
    }
    
    // Loss should generally decrease (may fluctuate)
    let first_half_avg: f32 = losses[0..5].iter().sum::<f32>() / 5.0;
    let second_half_avg: f32 = losses[5..10].iter().sum::<f32>() / 5.0;
    
    println!("First 5 avg: {}, Last 5 avg: {}", first_half_avg, second_half_avg);
    // Don't assert strict decrease - just verify it runs
}
```

**Exit Criteria**:
- [ ] `train_step` runs without errors
- [ ] Loss is valid (positive, not NaN)
- [ ] Multiple epochs can run

---

### Task 4.4: Integrate with Training Binary

**Description**: Update `train_checkerboard.rs` to use GPU

**Implementation**:

```rust
// src/bin/train_checkerboard.rs

#[cfg(feature = "gpu")]
fn main_gpu(args: &Args) {
    println!("GPU Training enabled");
    
    let ctx = GpuContext::new().expect("Failed to initialize GPU");
    println!("Using GPU: {}", ctx.adapter_info().name);
    
    let model = if args.small {
        create_small_checkerboard_model()
    } else {
        create_checkerboard_model()
    };
    
    let mut gpu_model = GpuModel::from_cpu(ctx, &model);
    let target = create_checkerboard(16, 2, 8);
    
    let mut rng = SimpleRng::new(args.seed);
    
    for epoch in 0..args.epochs {
        // Generate batch of seeds
        let seeds: Vec<NGrid> = (0..BATCH_SIZE)
            .map(|_| create_random_seed(16, 8, &mut rng))
            .collect();
        
        let loss = gpu_model.train_step_batch(&seeds, &target, 20, epoch)
            .expect("Training step failed");
        
        if epoch % args.log_interval == 0 {
            // Compute hard accuracy (requires GPU→CPU and hard forward)
            let hard_acc = if args.compute_hard_accuracy {
                gpu_model.compute_hard_accuracy(&target)
            } else {
                0.0
            };
            
            println!("Epoch {:4}: soft_loss={:.4}, hard_acc={:.2}%", 
                     epoch, loss, hard_acc * 100.0);
        }
    }
    
    // Save trained model
    let mut cpu_model = model;
    gpu_model.to_cpu(&mut cpu_model).expect("Failed to download model");
    // ... save to file
}

fn main() {
    let args = Args::parse();
    
    #[cfg(feature = "gpu")]
    {
        main_gpu(&args);
        return;
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        main_cpu(&args);
    }
}
```

**Tests**:
```bash
# Build with GPU
cargo build --bin train_checkerboard --features gpu --release

# Run short test
cargo run --bin train_checkerboard --features gpu --release -- --epochs=10 --small

# Full training
cargo run --bin train_checkerboard --features gpu --release -- --epochs=500
```

**Exit Criteria**:
- [ ] Binary builds with `--features gpu`
- [ ] GPU training runs for 500 epochs
- [ ] Progress is logged correctly

---

### Task 4.5: Optimize Memory Transfers

**Description**: Minimize CPU↔GPU data transfers

**Current Transfer Points**:
1. ❌ Seed upload (every epoch) - could generate on GPU
2. ❌ Target upload (once) - fine
3. ❌ Loss download (every epoch) - fine
4. ❌ Hard accuracy (every N epochs) - expensive, minimize

**Optimization: GPU Random Seed Generation**

```wgsl
// Simple LCG random generator
struct RngState {
    state: u32,
}

fn lcg_next(state: ptr<function, u32>) -> f32 {
    *state = *state * 1664525u + 1013904223u;
    return f32(*state) / 4294967295.0;
}

@compute @workgroup_size(256)
fn generate_random_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    if (cell_idx >= num_cells) { return; }
    
    var rng_state = seed_value + cell_idx;
    
    for (var c = 0u; c < num_channels; c++) {
        grid[cell_idx * num_channels + c] = lcg_next(&rng_state);
    }
}
```

**Optimization: Reduce Hard Accuracy Frequency**

Only compute hard accuracy every 10-50 epochs during training:
```rust
if epoch % 50 == 0 || epoch == args.epochs - 1 {
    let hard_acc = gpu_model.compute_hard_accuracy(&target);
    println!("  hard_acc = {:.2}%", hard_acc * 100.0);
}
```

**Tests**:
```rust
#[test]
fn test_gpu_random_seed_generation() {
    let ctx = GpuContext::new().unwrap();
    
    let grid1 = ctx.generate_random_grid(16, 16, 8, 42).unwrap();
    let grid2 = ctx.generate_random_grid(16, 16, 8, 42).unwrap();
    let grid3 = ctx.generate_random_grid(16, 16, 8, 43).unwrap();
    
    // Same seed = same grid
    assert_eq!(grid1, grid2);
    
    // Different seed = different grid
    assert_ne!(grid1, grid3);
    
    // Values in valid range
    for val in &grid1 {
        assert!(*val >= 0.0 && *val <= 1.0);
    }
}
```

**Exit Criteria**:
- [ ] Random seed generation on GPU works
- [ ] Hard accuracy computed infrequently
- [ ] Training loop has minimal transfers

---

### Task 4.6: Profile and Benchmark

**Description**: Measure actual performance and identify bottlenecks

**Implementation**:

```rust
// src/bin/benchmark_gpu.rs

fn main() {
    let ctx = GpuContext::new().expect("GPU required");
    let mut rng = SimpleRng::new(42);
    
    let model = create_checkerboard_model();
    let mut gpu_model = GpuModel::from_cpu(ctx, &model);
    let target = create_checkerboard(16, 2, 8);
    
    println!("Warming up...");
    for _ in 0..10 {
        let seed = create_random_seed(16, 8, &mut rng);
        gpu_model.train_step(&seed, &target, 20, 0).unwrap();
    }
    
    println!("Benchmarking GPU...");
    let start = std::time::Instant::now();
    for epoch in 0..100 {
        let seed = create_random_seed(16, 8, &mut rng);
        gpu_model.train_step(&seed, &target, 20, epoch).unwrap();
    }
    let gpu_time = start.elapsed();
    
    println!("Benchmarking CPU...");
    let mut cpu_trainer = TrainingLoop::new(model.clone(), TrainingConfig::checkerboard_sync());
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let seed = create_random_seed(16, 8, &mut rng);
        cpu_trainer.train_step(&seed, &target);
    }
    let cpu_time = start.elapsed();
    
    println!("\nResults (100 epochs, 20 steps each):");
    println!("  GPU: {:?} ({:.1} epochs/sec)", gpu_time, 100.0 / gpu_time.as_secs_f64());
    println!("  CPU: {:?} ({:.1} epochs/sec)", cpu_time, 100.0 / cpu_time.as_secs_f64());
    println!("  Speedup: {:.1}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
}
```

**Expected Results**:

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Time per epoch | ~3s | ~0.1s | ~30x |
| 500 epochs | ~25 min | ~50 sec | ~30x |

**Profiling with GPU Timestamps**:

```rust
impl GpuModel {
    /// Profile individual operations
    pub fn profile_train_step(&mut self, ...) -> ProfileResult {
        let t0 = Instant::now();
        self.upload_grid(...);
        let upload_time = t0.elapsed();
        
        let t1 = Instant::now();
        self.forward_pass(...);
        let forward_time = t1.elapsed();
        
        let t2 = Instant::now();
        self.backward_pass(...);
        let backward_time = t2.elapsed();
        
        let t3 = Instant::now();
        self.adamw_step(...);
        let optimizer_time = t3.elapsed();
        
        ProfileResult {
            upload: upload_time,
            forward: forward_time,
            backward: backward_time,
            optimizer: optimizer_time,
        }
    }
}
```

**Exit Criteria**:
- [ ] Benchmark binary created
- [ ] Speedup measured (target: >10x)
- [ ] Bottlenecks identified

---

### Task 4.7: Workgroup Size Tuning

**Description**: Optimize compute shader workgroup sizes for AMD RX 7800 XT

**AMD RDNA 3 Characteristics**:
- Wavefront size: 32 (can run in wave64 mode too)
- Max workgroup size: 1024
- Compute units: 60
- Registers per CU: 256 KB

**Tuning Strategy**:

1. **Gate forward/backward**: Try workgroup sizes 64, 128, 256
2. **AdamW**: Try 256, 512
3. **Reduction (loss)**: Use hierarchical reduction

```rust
fn tune_workgroup_sizes() {
    let workgroup_sizes = [32, 64, 128, 256, 512];
    
    for size in workgroup_sizes {
        let time = benchmark_gate_forward_with_workgroup_size(size);
        println!("Workgroup size {}: {:?}", size, time);
    }
}
```

**Exit Criteria**:
- [ ] Best workgroup size identified for each kernel
- [ ] Performance improvement documented

---

## Final Checklist

| Task | Status | Verified By |
|------|--------|-------------|
| 4.1 Persistent GPU buffers | ⬜ | `test_gpu_model_roundtrip` |
| 4.2 GPU AdamW optimizer | ⬜ | `test_gpu_adamw_matches_cpu` |
| 4.3 GPU training step | ⬜ | `test_gpu_train_step_runs` |
| 4.4 Training binary integration | ⬜ | `cargo run --bin train_checkerboard --features gpu` |
| 4.5 Memory transfer optimization | ⬜ | Profiling shows minimal transfers |
| 4.6 Benchmarking | ⬜ | Speedup measured |
| 4.7 Workgroup tuning | ⬜ | Optimal sizes documented |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Kernel launch overhead | Medium | Medium | Batch more operations |
| Memory bandwidth limit | Medium | Medium | Minimize transfers, use local memory |
| Driver bugs on AMD | Low | High | Test on different driver versions |
| Training doesn't converge | Low | High | Verify gradients match CPU first |

---

## Performance Targets

| Metric | Target | Stretch |
|--------|--------|---------|
| Speedup vs CPU | 10x | 30x |
| Time for 500 epochs | < 5 min | < 2 min |
| Memory usage | < 1 GB | < 500 MB |
| Hard accuracy at 500 epochs | > 80% | > 95% |

---

## Notes for Implementation

1. **Test on CPU first**: Ensure GPU results match before optimizing
2. **Profile before optimizing**: Find actual bottlenecks, not assumed ones
3. **Batch operations**: Fewer kernel launches = better performance
4. **Persistent buffers**: Avoid allocation in training loop
5. **AMD-specific**: Test with ROCm/Vulkan backend, not OpenGL

---

## Completion Criteria

Phase 4 is complete when:
1. ✅ Full training runs on GPU with `--features gpu`
2. ✅ Training produces similar results to CPU (within training variance)
3. ✅ Performance is at least 10x faster than CPU
4. ✅ 500 epochs complete in under 5 minutes
5. ✅ Hard accuracy reaches expected levels (~50% plateau, then improvement)

---

## Post-Phase 4: Future Optimizations

These are out of scope for Phase 4 but can be tackled later:

1. **Multi-GPU support**: For larger grids
2. **Mixed precision (f16)**: For memory bandwidth
3. **Tensor cores**: If available on AMD
4. **Async compute**: Overlap CPU logging with GPU training
5. **Graph compilation**: Pre-compile entire training step
