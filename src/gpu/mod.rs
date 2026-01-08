//! GPU acceleration module using wgpu.
//!
//! This module provides GPU-accelerated compute operations for training
//! differentiable logic cellular automata.

mod context;
mod gate_layer;
mod forward;

pub use context::{GpuContext, GpuError};

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_adapter_info() {
        // Skip if no GPU available
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let info = ctx.adapter_info();
        println!("Adapter: {}", info.name);
        println!("Backend: {:?}", info.backend);
        println!("Device type: {:?}", info.device_type);
    }

    #[test]
    fn test_double_kernel() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return, // Skip if no GPU
        };

        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let output = ctx.run_double_kernel(&input).expect("kernel failed");

        assert_eq!(output.len(), input.len());
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - input[i] * 2.0).abs() < 1e-6,
                "Mismatch at {}: {} != {}",
                i,
                v,
                input[i] * 2.0
            );
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
            assert!(
                (v - input[i] * 2.0).abs() < 1e-5,
                "Mismatch at {}: {} != {}",
                i,
                v,
                input[i] * 2.0
            );
        }
    }

    #[test]
    fn test_double_kernel_empty() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let input: Vec<f32> = vec![];
        let output = ctx.run_double_kernel(&input).expect("kernel failed");
        assert!(output.is_empty());
    }
}
