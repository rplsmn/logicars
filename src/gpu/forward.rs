//! GPU perception and update module forward pass.

use super::context::{GpuContext, GpuError};
use crate::grid::{NGrid, NNeighborhood};
use crate::perception::{PerceptionKernel, PerceptionModule};
use crate::update::{DiffLogicCA, UpdateModule};

impl GpuContext {
    /// Run a multi-layer kernel (PerceptionKernel or similar) forward on GPU.
    /// Processes layers sequentially, each layer on GPU.
    pub fn run_kernel_forward(&self, kernel: &PerceptionKernel, inputs: &[f64]) -> Result<Vec<f64>, GpuError> {
        let mut current: Vec<f64> = inputs.to_vec();

        for layer in &kernel.layers {
            current = self.run_gate_layer_forward(layer, &current)?;
        }

        Ok(current)
    }

    /// Run perception module forward on GPU for a single neighborhood.
    /// Returns (output, all_activations) matching CPU signature.
    pub fn run_perception_forward(
        &self,
        perception: &PerceptionModule,
        neighborhood: &NNeighborhood,
    ) -> Result<Vec<f64>, GpuError> {
        assert_eq!(neighborhood.channels, perception.channels);

        let kernel_output_size = perception.kernels[0].output_size();

        // kernel_outputs[k][c] = output values
        let mut kernel_outputs: Vec<Vec<Vec<f64>>> =
            vec![vec![Vec::new(); perception.channels]; perception.num_kernels];

        // Run all kernels on all channels
        for c in 0..perception.channels {
            let channel_inputs: Vec<f64> = (0..9)
                .map(|pos| neighborhood.get(pos, c))
                .collect();

            for (k, kernel) in perception.kernels.iter().enumerate() {
                let output = self.run_kernel_forward(kernel, &channel_inputs)?;
                kernel_outputs[k][c] = output;
            }
        }

        // Build output with correct ordering: (c s k)
        let mut output = Vec::with_capacity(perception.output_size());

        // First: center cell values
        output.extend_from_slice(neighborhood.center());

        // Then: for each channel, for each output bit, for each kernel
        for c in 0..perception.channels {
            for s in 0..kernel_output_size {
                for k in 0..perception.num_kernels {
                    output.push(kernel_outputs[k][c][s]);
                }
            }
        }

        Ok(output)
    }

    /// Run update module forward on GPU.
    pub fn run_update_forward(
        &self,
        update: &UpdateModule,
        inputs: &[f64],
    ) -> Result<Vec<f64>, GpuError> {
        let mut current = inputs.to_vec();

        for layer in &update.layers {
            current = self.run_gate_layer_forward(layer, &current)?;
        }

        Ok(current)
    }

    /// Run one CA step on GPU for the entire grid.
    /// Returns the output grid.
    pub fn run_ca_step(&self, model: &DiffLogicCA, grid: &NGrid) -> Result<NGrid, GpuError> {
        let mut output = NGrid::new(
            grid.width,
            grid.height,
            grid.channels,
            grid.boundary.clone(),
        );

        for y in 0..grid.height {
            for x in 0..grid.width {
                let neighborhood = grid.neighborhood(x, y);

                // GPU perception
                let perception_out = self.run_perception_forward(&model.perception, &neighborhood)?;

                // GPU update
                let update_out = self.run_update_forward(&model.update, &perception_out)?;

                // Write to output grid
                for c in 0..grid.channels {
                    output.set(x, y, c, update_out[c]);
                }
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkerboard::{create_checkerboard_perception, create_checkerboard_update, create_random_seed};
    use crate::training::SimpleRng;

    #[test]
    fn test_perception_gpu_vs_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut rng = SimpleRng::new(999);
        let perception = create_checkerboard_perception();
        let grid = create_random_seed(16, 8, &mut rng);

        // Test a few cells
        for (x, y) in [(0, 0), (7, 7), (15, 15)] {
            let neighborhood = grid.neighborhood(x, y);

            let (cpu_output, _) = perception.forward_soft(&neighborhood);
            let gpu_output = ctx.run_perception_forward(&perception, &neighborhood).unwrap();

            assert_eq!(cpu_output.len(), gpu_output.len(), "Output size mismatch");

            let max_diff: f64 = cpu_output
                .iter()
                .zip(gpu_output.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0, f64::max);

            assert!(
                max_diff < 1e-4,
                "Cell ({},{}): max diff = {}",
                x,
                y,
                max_diff
            );
        }
    }

    #[test]
    fn test_update_gpu_vs_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut rng = SimpleRng::new(777);
        let update = create_checkerboard_update();

        // Random input matching perception output size (264)
        let input: Vec<f64> = (0..264).map(|_| rng.next_f64()).collect();

        let cpu_activations = update.forward_soft(&input);
        let cpu_output = cpu_activations.last().cloned().unwrap();
        let gpu_output = ctx.run_update_forward(&update, &input).unwrap();

        assert_eq!(cpu_output.len(), gpu_output.len());
        assert_eq!(cpu_output.len(), 8); // 8 output channels

        let max_diff: f64 = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0, f64::max);

        assert!(max_diff < 1e-4, "max diff = {}", max_diff);
    }

    #[test]
    fn test_perception_output_size() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut rng = SimpleRng::new(123);
        let perception = create_checkerboard_perception();
        let grid = create_random_seed(16, 8, &mut rng);
        let neighborhood = grid.neighborhood(0, 0);

        let gpu_output = ctx.run_perception_forward(&perception, &neighborhood).unwrap();

        // Expected: 8 (center) + 16 kernels * 2 output bits * 8 channels = 8 + 256 = 264
        assert_eq!(gpu_output.len(), 264, "Perception output size should be 264");
    }

    #[test]
    #[ignore] // Slow: naive per-cell dispatch. Run with --ignored flag.
    fn test_ca_step_gpu_vs_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut rng = SimpleRng::new(555);
        let model = crate::checkerboard::create_small_checkerboard_model();
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

        // GPU forward
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

        assert!(max_diff < 1e-4, "CA step max diff = {}", max_diff);
    }

    #[test]
    #[ignore] // Slow: naive per-cell dispatch. Run with --ignored flag.
    fn test_ca_step_performance() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut rng = SimpleRng::new(111);
        let model = crate::checkerboard::create_small_checkerboard_model();
        let grid = create_random_seed(16, 8, &mut rng);

        // Warmup
        for _ in 0..2 {
            let _ = ctx.run_ca_step(&model, &grid);
        }

        // Benchmark GPU
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = ctx.run_ca_step(&model, &grid);
        }
        let gpu_time = start.elapsed();

        // Benchmark CPU
        let start = std::time::Instant::now();
        for _ in 0..10 {
            for y in 0..grid.height {
                for x in 0..grid.width {
                    let neighborhood = grid.neighborhood(x, y);
                    let _ = model.forward_soft(&neighborhood);
                }
            }
        }
        let cpu_time = start.elapsed();

        println!("GPU: {:?} for 10 steps", gpu_time);
        println!("CPU: {:?} for 10 steps", cpu_time);
        println!(
            "Speedup: {:.2}x",
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        );
    }
}
