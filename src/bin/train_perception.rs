//! Phase 1.1 Integration Test: Perception Kernel Training
//!
//! Tests the perception kernel's ability to learn Game of Life rules
//! on all 512 neighborhood configurations.
//!
//! Exit criteria: >95% accuracy

use logicars::{GolTruthTable, PerceptionKernel, PerceptionTrainer, PerceptionTopology, DeepPerceptionKernel, DeepPerceptionTrainer};

fn main() {
    println!("=== Phase 1.1: Perception Kernel Training ===\n");

    let truth_table = GolTruthTable::new();

    // Count expected alive states
    let alive_count = truth_table.targets.iter().filter(|&&t| t).count();
    println!("Game of Life truth table statistics:");
    println!("  Total configurations: 512");
    println!("  Alive outcomes: {}", alive_count);
    println!("  Dead outcomes: {}", 512 - alive_count);
    println!();

    // Test 1: Minimal kernel (simpler, faster training)
    println!("--- Test 1: Minimal Kernel (4 gates in layer 1) ---");
    let kernel = PerceptionKernel::with_topology(PerceptionTopology::minimal());
    println!("Kernel has {} total gates", kernel.num_gates());

    let mut trainer = PerceptionTrainer::new(kernel, 0.05);
    let result = trainer.train(&truth_table, 5000, 1e-4, true);

    println!("\nTest 1 Results:");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Final loss: {:.6}", result.final_loss);
    println!("  Hard accuracy: {:.2}%", result.hard_accuracy * 100.0);
    println!("  Exit criteria met: {}", result.meets_exit_criteria());
    println!();

    // Test 2: Full first_kernel topology (16 gates in layer 1)
    println!("--- Test 2: Full Kernel (16 gates in layer 1) ---");
    let kernel = PerceptionKernel::new();
    println!("Kernel has {} total gates", kernel.num_gates());

    let mut trainer = PerceptionTrainer::new(kernel, 0.05);
    let result = trainer.train(&truth_table, 10000, 1e-4, true);

    println!("\nTest 2 Results:");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Final loss: {:.6}", result.final_loss);
    println!("  Hard accuracy: {:.2}%", result.hard_accuracy * 100.0);
    println!("  Exit criteria met: {}", result.meets_exit_criteria());
    println!();

    // Test 3: Deep Perception Kernel (recommended architecture)
    println!("--- Test 3: Deep Perception Kernel (6 layers, 63 gates) ---");
    let kernel = DeepPerceptionKernel::new();
    println!("Kernel has {} total gates across {} layers", kernel.num_gates(), 6);

    let mut trainer = DeepPerceptionTrainer::new(kernel, 0.05);
    let result = trainer.train(&truth_table, 10000, 1e-4, true);

    println!("\nTest 3 Results:");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Final loss: {:.6}", result.final_loss);
    println!("  Hard accuracy: {:.2}%", result.hard_accuracy * 100.0);
    println!("  Exit criteria met: {}", result.meets_exit_criteria());
    println!();

    // Test 4: Deep kernel with lower learning rate if not converged
    if !result.meets_exit_criteria() {
        println!("--- Test 4: Deep Kernel with Lower LR (extended training) ---");
        let kernel = DeepPerceptionKernel::new();
        let mut trainer = DeepPerceptionTrainer::new(kernel, 0.02);
        let result = trainer.train(&truth_table, 20000, 1e-5, true);

        println!("\nTest 4 Results:");
        println!("  Converged: {}", result.converged);
        println!("  Iterations: {}", result.iterations);
        println!("  Final loss: {:.6}", result.final_loss);
        println!("  Hard accuracy: {:.2}%", result.hard_accuracy * 100.0);
        println!("  Exit criteria met: {}", result.meets_exit_criteria());
    }

    // Final summary
    println!("\n=== Summary ===");
    if result.meets_exit_criteria() {
        println!("SUCCESS: Exit criteria met (>95% accuracy)");
    } else {
        println!("WARNING: Exit criteria not yet met");
        println!("Consider: larger network, more iterations, or architecture changes");
    }
}
