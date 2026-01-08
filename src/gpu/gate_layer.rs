//! GPU gate layer forward pass implementation.

use super::context::{GpuContext, GpuError};
use crate::perception::GateLayer;
use wgpu::util::DeviceExt;

/// Uniform buffer layout for layer configuration
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerConfig {
    num_gates: u32,
    input_offset: u32,
    output_offset: u32,
    _padding: u32,
}

impl GpuContext {
    /// Run a gate layer forward pass on GPU.
    ///
    /// Takes inputs as f64 (CPU format), runs on GPU in f32, returns f64.
    pub fn run_gate_layer_forward(
        &self,
        layer: &GateLayer,
        inputs: &[f64],
    ) -> Result<Vec<f64>, GpuError> {
        // Convert f64 to f32 for GPU
        let inputs_f32: Vec<f32> = inputs.iter().map(|&x| x as f32).collect();
        let logits_f32 = layer.get_logits_flat_f32();
        let wires_u32 = layer.get_wires_flat_u32();

        let output_f32 = self.run_gate_forward_internal(
            &inputs_f32,
            &logits_f32,
            &wires_u32,
            layer.output_size(),
        )?;

        // Convert back to f64
        Ok(output_f32.iter().map(|&x| x as f64).collect())
    }

    /// Internal gate forward pass on GPU with f32 buffers.
    pub(crate) fn run_gate_forward_internal(
        &self,
        inputs: &[f32],
        logits: &[f32],
        wires: &[u32],
        num_gates: usize,
    ) -> Result<Vec<f32>, GpuError> {
        if num_gates == 0 {
            return Ok(vec![]);
        }

        let config = LayerConfig {
            num_gates: num_gates as u32,
            input_offset: 0,
            output_offset: 0,
            _padding: 0,
        };

        // Create buffers
        let config_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gate_config"),
                contents: bytemuck::bytes_of(&config),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gate_inputs"),
                contents: bytemuck::cast_slice(inputs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let logits_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gate_logits"),
                contents: bytemuck::cast_slice(logits),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let wires_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gate_wires"),
                contents: bytemuck::cast_slice(wires),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gate_outputs"),
            size: (num_gates * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gate_staging"),
            size: (num_gates * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gate_forward_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/gate_forward.wgsl").into(),
                ),
            });

        // Create compute pipeline
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gate_forward_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gate_forward_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: logits_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wires_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gate_forward_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gate_forward_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (num_gates as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (num_gates * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| GpuError::BufferMap(e.to_string()))?
            .map_err(|e| GpuError::BufferMap(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::{generate_connections, ConnectionType};
    use crate::training::SimpleRng;

    #[test]
    fn test_gate_forward_single_gate_passthrough() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return, // Skip if no GPU
        };

        // Single gate with pass-through initialization (logits[3] = 10.0)
        let mut logits = vec![0.0f32; 16];
        logits[3] = 10.0; // A (pass-through)

        let inputs = vec![0.7f32, 0.3f32];
        let wires = vec![0u32, 1u32];

        let output = ctx
            .run_gate_forward_internal(&inputs, &logits, &wires, 1)
            .unwrap();

        // Should be very close to 0.7 (input a) since pass-through dominates
        assert!(
            (output[0] - 0.7).abs() < 0.01,
            "Expected ~0.7, got {}",
            output[0]
        );
    }

    #[test]
    fn test_gate_forward_single_gate_and() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        // Single gate with AND operation dominant (logits[1] = 10.0)
        let mut logits = vec![0.0f32; 16];
        logits[1] = 10.0; // And

        let inputs = vec![0.8f32, 0.6f32];
        let wires = vec![0u32, 1u32];

        let output = ctx
            .run_gate_forward_internal(&inputs, &logits, &wires, 1)
            .unwrap();

        // Should be close to 0.8 * 0.6 = 0.48
        assert!(
            (output[0] - 0.48).abs() < 0.01,
            "Expected ~0.48, got {}",
            output[0]
        );
    }

    #[test]
    fn test_gate_layer_gpu_vs_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut rng = SimpleRng::new(42);

        // Create a layer with 4 inputs, 2 outputs
        let wires = generate_connections(ConnectionType::Unique, 4, 2);
        let layer = GateLayer::new(2, wires);

        let inputs = vec![0.2, 0.4, 0.6, 0.8];

        // CPU forward
        let cpu_output = layer.forward_soft(&inputs);

        // GPU forward
        let gpu_output = ctx.run_gate_layer_forward(&layer, &inputs).unwrap();

        // Compare
        assert_eq!(cpu_output.len(), gpu_output.len());
        for (i, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
            assert!(
                (cpu - gpu).abs() < 1e-5,
                "Mismatch at {}: CPU={}, GPU={}",
                i,
                cpu,
                gpu
            );
        }
    }

    #[test]
    fn test_gate_layer_gpu_vs_cpu_larger() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut rng = SimpleRng::new(12345);

        // Test various layer sizes
        for (input_size, output_size) in [(8, 4), (16, 8), (64, 32), (256, 128)] {
            let wires = generate_connections(ConnectionType::Unique, input_size, output_size);
            let layer = GateLayer::new(output_size, wires);
            let inputs: Vec<f64> = (0..input_size).map(|_| rng.next_f64()).collect();

            let cpu = layer.forward_soft(&inputs);
            let gpu = ctx.run_gate_layer_forward(&layer, &inputs).unwrap();

            let max_diff: f64 = cpu
                .iter()
                .zip(gpu.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0, f64::max);

            assert!(
                max_diff < 1e-5,
                "Layer {}→{}: max diff = {}",
                input_size,
                output_size,
                max_diff
            );
        }
    }
}
