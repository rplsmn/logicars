//! Test binary for GPU functionality.
//!
//! Run with: cargo run --bin test_gpu --features gpu --release

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("GPU feature not enabled. Run with: cargo run --bin test_gpu --features gpu");
        std::process::exit(1);
    }

    #[cfg(feature = "gpu")]
    {
        run_gpu_tests();
    }
}

#[cfg(feature = "gpu")]
fn run_gpu_tests() {
    use logicars::gpu::GpuContext;

    println!("=== GPU Phase 1 Verification ===\n");

    // 1. Initialize GPU context
    println!("Initializing GPU...");
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("✗ Failed to create GPU context: {}", e);
            std::process::exit(1);
        }
    };

    // 2. Print adapter info
    let info = ctx.adapter_info();
    println!("✓ GPU: {} ({:?})", info.name, info.backend);
    println!("  Device type: {:?}", info.device_type);
    println!("  Driver: {}", info.driver);
    println!("  Driver info: {}", info.driver_info);

    // 3. Run small test kernel
    println!("\nRunning test kernel (small)...");
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let output = match ctx.run_double_kernel(&input) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("✗ Kernel failed: {}", e);
            std::process::exit(1);
        }
    };

    let small_correct = output
        .iter()
        .zip(input.iter())
        .all(|(o, i)| (o - i * 2.0).abs() < 1e-5);
    if small_correct {
        println!("✓ Small test passed: {:?} → {:?}", input, output);
    } else {
        eprintln!("✗ Small test failed: {:?} → {:?}", input, output);
        std::process::exit(1);
    }

    // 4. Run large test kernel
    println!("\nRunning test kernel (large, 100K elements)...");
    let start = std::time::Instant::now();
    let large_input: Vec<f32> = (0..100_000).map(|i| i as f32).collect();
    let large_output = match ctx.run_double_kernel(&large_input) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("✗ Large kernel failed: {}", e);
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();

    let large_correct = large_output
        .iter()
        .zip(large_input.iter())
        .all(|(o, i)| (o - i * 2.0).abs() < 1e-4);
    if large_correct {
        println!(
            "✓ Large test passed: {} elements in {:?}",
            large_input.len(),
            elapsed
        );
    } else {
        // Find first mismatch
        for (i, (o, i_val)) in large_output.iter().zip(large_input.iter()).enumerate() {
            if (o - i_val * 2.0).abs() >= 1e-4 {
                eprintln!(
                    "✗ Large test failed at index {}: {} != {} * 2",
                    i, o, i_val
                );
                break;
            }
        }
        std::process::exit(1);
    }

    // 5. Summary
    println!("\n=== GPU Phase 1 Complete ===");
    println!("✓ GPU context creation works");
    println!("✓ Compute shader compilation works");
    println!("✓ Buffer read/write roundtrip verified");
    println!("✓ All tests passed!");
}
