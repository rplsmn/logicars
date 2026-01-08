//! Batched GPU forward pass - processes all cells in parallel.

use super::context::{GpuContext, GpuError};
use crate::grid::{NGrid, NNeighborhood};
use crate::perception::GateLayer;
use crate::update::DiffLogicCA;
use wgpu::util::DeviceExt;

/// Uniform buffer layout for batched layer configuration
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchedLayerConfig {
    num_gates: u32,
    num_cells: u32,
    input_stride: u32,
    output_stride: u32,
}

impl GpuContext {
    /// Run a gate layer forward pass on GPU for multiple cells in parallel.
    ///
    /// inputs: [cell_0_inputs, cell_1_inputs, ...] flattened
    /// Returns: [cell_0_outputs, cell_1_outputs, ...] flattened
    pub fn run_gate_layer_batched(
        &self,
        layer: &GateLayer,
        inputs: &[f32],
        num_cells: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let input_stride = inputs.len() / num_cells;
        let output_stride = layer.output_size();
        let num_gates = layer.num_gates();

        if num_cells == 0 || num_gates == 0 {
            return Ok(vec![]);
        }

        let config = BatchedLayerConfig {
            num_gates: num_gates as u32,
            num_cells: num_cells as u32,
            input_stride: input_stride as u32,
            output_stride: output_stride as u32,
        };

        let logits_f32 = layer.get_logits_flat_f32();
        let wires_u32 = layer.get_wires_flat_u32();

        // Create buffers
        let config_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batched_config"),
                contents: bytemuck::bytes_of(&config),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batched_inputs"),
                contents: bytemuck::cast_slice(inputs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let logits_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batched_logits"),
                contents: bytemuck::cast_slice(&logits_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let wires_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batched_wires"),
                contents: bytemuck::cast_slice(&wires_u32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = num_cells * output_stride;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batched_outputs"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batched_staging"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("batched_gate_forward_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/gate_forward_batched.wgsl").into(),
                ),
            });

        // Create compute pipeline
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("batched_gate_forward_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batched_gate_forward_bind_group"),
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
                label: Some("batched_gate_forward_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("batched_gate_forward_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch: ceil(num_gates/64) × ceil(num_cells/4)
            let workgroups_x = (num_gates as u32 + 63) / 64;
            let workgroups_y = (num_cells as u32 + 3) / 4;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (output_size * std::mem::size_of::<f32>()) as u64,
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

    /// Run a full CA step on GPU with batched processing.
    /// All cells are processed in parallel for each layer.
    pub fn run_ca_step_batched(&self, model: &DiffLogicCA, grid: &NGrid) -> Result<NGrid, GpuError> {
        let num_cells = grid.width * grid.height;
        let channels = grid.channels;

        // Step 1: Prepare all neighborhoods as perception inputs
        // For each cell, we need 9 values per channel for each kernel
        // But kernels share the same 9-cell input per channel
        
        // Collect all neighborhoods
        let mut neighborhoods: Vec<NNeighborhood> = Vec::with_capacity(num_cells);
        for y in 0..grid.height {
            for x in 0..grid.width {
                neighborhoods.push(grid.neighborhood(x, y));
            }
        }

        // Step 2: Run perception for all cells
        // For each kernel, for each channel, run the kernel layers on all cells
        let perception = &model.perception;
        let kernel_output_size = perception.kernels[0].output_size();
        
        // perception_outputs[cell_idx] = full perception output (264 values for checkerboard)
        let mut perception_outputs: Vec<Vec<f64>> = vec![Vec::new(); num_cells];
        
        // Initialize with center cell values
        for (cell_idx, neighborhood) in neighborhoods.iter().enumerate() {
            perception_outputs[cell_idx] = neighborhood.center().to_vec();
        }

        // kernel_results[k][c][cell_idx] = output values for kernel k, channel c, cell cell_idx
        let mut kernel_results: Vec<Vec<Vec<Vec<f64>>>> = 
            vec![vec![vec![Vec::new(); num_cells]; channels]; perception.num_kernels];

        // Process each kernel
        for (k, kernel) in perception.kernels.iter().enumerate() {
            // Process each channel
            for c in 0..channels {
                // Prepare inputs for all cells: 9 values per cell
                let mut all_inputs: Vec<f32> = Vec::with_capacity(num_cells * 9);
                for neighborhood in &neighborhoods {
                    for pos in 0..9 {
                        all_inputs.push(neighborhood.get(pos, c) as f32);
                    }
                }

                // Run through kernel layers
                let mut current = all_inputs;
                for layer in &kernel.layers {
                    let input_stride = current.len() / num_cells;
                    current = self.run_gate_layer_batched(layer, &current, num_cells)?;
                }

                // Store results: current is [cell_0_output, cell_1_output, ...]
                let output_size = kernel_output_size;
                for cell_idx in 0..num_cells {
                    let start = cell_idx * output_size;
                    let end = start + output_size;
                    kernel_results[k][c][cell_idx] = current[start..end]
                        .iter()
                        .map(|&x| x as f64)
                        .collect();
                }
            }
        }

        // Build perception outputs with correct ordering: (c s k)
        for cell_idx in 0..num_cells {
            for c in 0..channels {
                for s in 0..kernel_output_size {
                    for k in 0..perception.num_kernels {
                        perception_outputs[cell_idx].push(kernel_results[k][c][cell_idx][s]);
                    }
                }
            }
        }

        // Step 3: Run update for all cells
        // Prepare all perception outputs as update inputs
        let update = &model.update;
        let mut current: Vec<f32> = perception_outputs
            .iter()
            .flat_map(|p| p.iter().map(|&x| x as f32))
            .collect();

        // Run through update layers
        for layer in &update.layers {
            current = self.run_gate_layer_batched(layer, &current, num_cells)?;
        }

        // Step 4: Build output grid
        let mut output = NGrid::new(grid.width, grid.height, channels, grid.boundary.clone());
        let output_channels = update.output_channels;
        
        for cell_idx in 0..num_cells {
            let x = cell_idx % grid.width;
            let y = cell_idx / grid.width;
            let start = cell_idx * output_channels;
            for c in 0..channels {
                output.set(x, y, c, current[start + c] as f64);
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkerboard::{create_random_seed, create_small_checkerboard_model};
    use crate::training::SimpleRng;

    #[test]
    fn test_batched_gate_layer() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        use crate::perception::{generate_connections, ConnectionType};

        // Create a simple layer
        let wires = generate_connections(ConnectionType::Unique, 4, 2);
        let layer = GateLayer::new(2, wires);

        // Prepare inputs for 3 cells
        let num_cells = 3;
        let inputs: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, // cell 0
            0.5, 0.6, 0.7, 0.8, // cell 1
            0.9, 0.2, 0.1, 0.4, // cell 2
        ];

        let batched_output = ctx.run_gate_layer_batched(&layer, &inputs, num_cells).unwrap();

        // Verify against individual forward passes
        for cell_idx in 0..num_cells {
            let cell_input: Vec<f64> = inputs[cell_idx * 4..(cell_idx + 1) * 4]
                .iter()
                .map(|&x| x as f64)
                .collect();
            let expected = layer.forward_soft(&cell_input);

            let start = cell_idx * 2;
            for (i, &exp) in expected.iter().enumerate() {
                let got = batched_output[start + i] as f64;
                assert!(
                    (exp - got).abs() < 1e-5,
                    "Cell {} output {}: expected {}, got {}",
                    cell_idx,
                    i,
                    exp,
                    got
                );
            }
        }
    }

    #[test]
    fn test_ca_step_batched_vs_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut rng = SimpleRng::new(555);
        let model = create_small_checkerboard_model();
        let grid = create_random_seed(16, 8, &mut rng);

        // CPU forward
        let mut cpu_output = NGrid::new(grid.width, grid.height, grid.channels, grid.boundary.clone());
        for y in 0..grid.height {
            for x in 0..grid.width {
                let neighborhood = grid.neighborhood(x, y);
                let (output, _, _) = model.forward_soft(&neighborhood);
                for c in 0..grid.channels {
                    cpu_output.set(x, y, c, output[c]);
                }
            }
        }

        // GPU batched forward
        let gpu_output = ctx.run_ca_step_batched(&model, &grid).unwrap();

        // Compare
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

        assert!(max_diff < 1e-4, "CA step batched max diff = {}", max_diff);
    }

    #[test]
    fn test_ca_step_batched_performance() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut rng = SimpleRng::new(111);
        let model = create_small_checkerboard_model();
        let grid = create_random_seed(16, 8, &mut rng);

        // Warmup
        for _ in 0..3 {
            let _ = ctx.run_ca_step_batched(&model, &grid);
        }

        // Benchmark GPU batched
        let start = std::time::Instant::now();
        for _ in 0..20 {
            let _ = ctx.run_ca_step_batched(&model, &grid);
        }
        let gpu_time = start.elapsed();

        // Benchmark CPU
        let start = std::time::Instant::now();
        for _ in 0..20 {
            for y in 0..grid.height {
                for x in 0..grid.width {
                    let neighborhood = grid.neighborhood(x, y);
                    let _ = model.forward_soft(&neighborhood);
                }
            }
        }
        let cpu_time = start.elapsed();

        println!("GPU batched: {:?} for 20 steps", gpu_time);
        println!("CPU: {:?} for 20 steps", cpu_time);
        println!(
            "Speedup: {:.2}x",
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        );

        // We expect GPU to be faster with batching
        // But don't assert - just report for now
    }
}
