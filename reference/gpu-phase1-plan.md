# GPU Phase 1: Basic wgpu Setup

## Overview

**Goal**: Establish wgpu infrastructure and verify AMD RX 7800 XT GPU works with a simple test kernel.

**Estimated Duration**: 1-2 days

**Dependencies**: None (foundational phase)

---

## Success Criteria

1. ✅ wgpu dependency added and compiles
2. ✅ GPU device enumeration works (finds AMD RX 7800 XT)
3. ✅ Simple compute shader executes successfully
4. ✅ Buffer read/write roundtrip verified
5. ✅ Unit tests for GPU context creation pass

---

## Task Breakdown

### Task 1.1: Add wgpu Dependencies

**Description**: Add required GPU dependencies to Cargo.toml

**Implementation**:
```toml
[dependencies]
wgpu = "24.0"
pollster = "0.4"  # Blocking on async GPU operations
bytemuck = { version = "1.18", features = ["derive"] }  # GPU buffer casting
```

**Tests**:
- `cargo build` succeeds with new dependencies

**Exit Criteria**:
- [ ] Dependencies added to Cargo.toml
- [ ] `cargo build` succeeds
- [ ] No version conflicts with existing dependencies

---

### Task 1.2: Create GPU Module Structure

**Description**: Create the gpu module with basic file structure

**Implementation**:
```
src/
├── gpu/
│   ├── mod.rs          # Module exports
│   ├── context.rs      # GPU device/queue/adapter
│   ├── buffer.rs       # Buffer management utilities
│   └── shaders/        # WGSL shader files (optional, can embed)
```

**Files to Create**:
1. `src/gpu/mod.rs` - Module root with public exports
2. `src/gpu/context.rs` - `GpuContext` struct holding device, queue, adapter

**Exit Criteria**:
- [ ] `src/gpu/` directory exists
- [ ] `mod.rs` declares submodules
- [ ] `context.rs` has `GpuContext` struct skeleton
- [ ] Module compiles (even if empty implementations)

---

### Task 1.3: Implement GpuContext

**Description**: Create `GpuContext` struct for managing GPU resources

**Implementation**:

```rust
// src/gpu/context.rs

use wgpu::{Adapter, Device, Instance, Queue};

pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    adapter: Adapter,
}

impl GpuContext {
    /// Create GPU context, preferring high-performance GPU
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }
    
    async fn new_async() -> Result<Self, GpuError> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;
        
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;
        
        Ok(Self { device, queue, adapter })
    }
    
    /// Get adapter info for debugging
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }
}

#[derive(Debug, Clone)]
pub enum GpuError {
    NoAdapter,
    DeviceRequest(String),
    BufferMap(String),
    ShaderCompile(String),
}
```

**Tests**:
```rust
#[test]
fn test_gpu_context_creation() {
    let ctx = GpuContext::new();
    // May fail on CI without GPU, so just check it doesn't panic unexpectedly
    if let Ok(ctx) = ctx {
        let info = ctx.adapter_info();
        println!("GPU: {} ({:?})", info.name, info.backend);
        assert!(!info.name.is_empty());
    }
}

#[test]
fn test_adapter_is_amd() {
    // Only run locally, skip in CI
    if std::env::var("CI").is_ok() {
        return;
    }
    let ctx = GpuContext::new().expect("GPU required for this test");
    let info = ctx.adapter_info();
    // Verify we're getting the RX 7800 XT (or at least AMD)
    assert!(
        info.name.contains("AMD") || info.name.contains("Radeon"),
        "Expected AMD GPU, got: {}",
        info.name
    );
}
```

**Exit Criteria**:
- [ ] `GpuContext::new()` compiles
- [ ] Test runs locally and finds GPU
- [ ] Adapter info shows AMD RX 7800 XT (or similar)

---

### Task 1.4: Implement Simple Test Compute Shader

**Description**: Create a minimal compute shader that doubles input values to verify GPU compute works.

**Implementation**:

**WGSL Shader** (`src/gpu/shaders/double.wgsl` or embedded string):
```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&input)) {
        output[idx] = input[idx] * 2.0;
    }
}
```

**Rust Code** (`src/gpu/compute.rs`):
```rust
use wgpu::{Buffer, BufferUsages, ShaderModule};
use bytemuck::{Pod, Zeroable};

impl GpuContext {
    /// Run a simple compute shader that doubles input values
    pub fn run_double_kernel(&self, input: &[f32]) -> Result<Vec<f32>, GpuError> {
        // 1. Create input buffer
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytemuck::cast_slice(input),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        
        // 2. Create output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (input.len() * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // 3. Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (input.len() * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // 4. Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("double_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/double.wgsl").into()),
        });
        
        // 5. Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("double_pipeline"),
            layout: None, // Auto layout
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // 6. Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("double_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        
        // 7. Create command encoder and dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("double_encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("double_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (input.len() as u32 + 63) / 64; // Ceil division
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // 8. Copy output to staging
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, 
            (input.len() * std::mem::size_of::<f32>()) as u64);
        
        self.queue.submit(Some(encoder.finish()));
        
        // 9. Read back results
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        
        Ok(result)
    }
}
```

**Tests**:
```rust
#[test]
fn test_double_kernel() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(_) => return, // Skip if no GPU
    };
    
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let output = ctx.run_double_kernel(&input).expect("kernel failed");
    
    assert_eq!(output.len(), input.len());
    for (i, &v) in output.iter().enumerate() {
        assert!((v - input[i] * 2.0).abs() < 1e-6, "Mismatch at {}: {} != {}", i, v, input[i] * 2.0);
    }
}

#[test]
fn test_double_kernel_large() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    
    let input: Vec<f32> = (0..10000).map(|i| i as f32).collect();
    let output = ctx.run_double_kernel(&input).expect("kernel failed");
    
    assert_eq!(output.len(), input.len());
    for (i, &v) in output.iter().enumerate() {
        assert!((v - input[i] * 2.0).abs() < 1e-5);
    }
}
```

**Exit Criteria**:
- [ ] WGSL shader compiles without errors
- [ ] Compute pipeline created successfully
- [ ] `test_double_kernel` passes (values doubled correctly)
- [ ] `test_double_kernel_large` passes (10K elements)

---

### Task 1.5: Create GPU Feature Flag

**Description**: Make GPU code optional via Cargo feature flag so it doesn't break builds without GPU.

**Implementation**:
```toml
# Cargo.toml
[features]
default = []
gpu = ["dep:wgpu", "dep:pollster"]

[dependencies]
wgpu = { version = "24.0", optional = true }
pollster = { version = "0.4", optional = true }
bytemuck = { version = "1.18", features = ["derive"] }  # Always needed
```

```rust
// src/lib.rs
#[cfg(feature = "gpu")]
pub mod gpu;
```

**Tests**:
```bash
# Without GPU feature
cargo build
cargo test --lib

# With GPU feature  
cargo build --features gpu
cargo test --lib --features gpu
```

**Exit Criteria**:
- [ ] `cargo build` works without GPU feature
- [ ] `cargo build --features gpu` enables GPU module
- [ ] All existing tests still pass

---

### Task 1.6: Create GPU Test Binary

**Description**: Create a simple binary to test GPU interactively

**Implementation** (`src/bin/test_gpu.rs`):
```rust
#[cfg(feature = "gpu")]
use logicars::gpu::GpuContext;

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("GPU feature not enabled. Run with: cargo run --bin test_gpu --features gpu");
        return;
    }
    
    #[cfg(feature = "gpu")]
    {
        println!("Initializing GPU...");
        let ctx = GpuContext::new().expect("Failed to create GPU context");
        
        let info = ctx.adapter_info();
        println!("✓ GPU: {} ({:?})", info.name, info.backend);
        println!("  Driver: {}", info.driver);
        
        println!("\nRunning test kernel...");
        let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let output = ctx.run_double_kernel(&input).expect("Kernel failed");
        
        let correct = output.iter().zip(input.iter()).all(|(o, i)| (o - i * 2.0).abs() < 1e-5);
        if correct {
            println!("✓ Compute kernel works correctly");
        } else {
            println!("✗ Compute kernel produced incorrect results");
        }
        
        println!("\n✓ GPU Phase 1 complete!");
    }
}
```

**Exit Criteria**:
- [ ] `cargo run --bin test_gpu --features gpu --release` runs successfully
- [ ] Output shows AMD GPU name
- [ ] Compute kernel verification passes

---

## Final Checklist

| Task | Status | Verified By |
|------|--------|-------------|
| 1.1 Dependencies added | ⬜ | `cargo build --features gpu` |
| 1.2 Module structure created | ⬜ | Files exist |
| 1.3 GpuContext implemented | ⬜ | `test_gpu_context_creation` |
| 1.4 Test compute shader works | ⬜ | `test_double_kernel` |
| 1.5 Feature flag works | ⬜ | Both feature on/off compile |
| 1.6 Test binary works | ⬜ | `cargo run --bin test_gpu --features gpu` |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| AMD drivers not compatible | Low | High | Use Vulkan backend, verify ROCm installed |
| wgpu version conflicts | Low | Medium | Pin exact versions |
| Shader compilation errors | Medium | Low | Start with minimal shader, iterate |
| Buffer mapping issues | Medium | Medium | Use staging buffer pattern |

---

## Notes for Implementation

1. **Always use Vulkan backend for AMD** - OpenGL may have issues
2. **Poll device after submit** - Required for synchronization
3. **Use staging buffers for readback** - Direct map not always supported
4. **Test on actual hardware** - GPU tests may fail in CI

---

## Next Phase

After Phase 1 is complete, proceed to **GPU Phase 2: Forward Pass on GPU** which will implement the actual gate forward computation shaders.
