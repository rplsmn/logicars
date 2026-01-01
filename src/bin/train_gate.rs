//! Phase 0.1: Single Gate Training Test
//!
//! This binary trains a single probabilistic gate to learn AND, OR, and XOR operations.
//! Exit criteria: >99% accuracy on truth tables, reproducible convergence.

use logicars::{BinaryOp, GateTrainer, TruthTable};

fn main() {
    println!("=== Phase 0.1: Single Gate Training ===\n");
    println!("Testing convergence on AND, OR, and XOR operations");
    println!("Exit criteria: >99% hard accuracy, loss converges\n");

    // Test AND
    println!("--- Training AND gate ---");
    test_operation(BinaryOp::And, "AND");

    println!("\n--- Training OR gate ---");
    test_operation(BinaryOp::Or, "OR");

    println!("\n--- Training XOR gate ---");
    test_operation(BinaryOp::Xor, "XOR");

    println!("\n=== Phase 0.1 Complete ===");
    println!("All gates trained successfully!");
}

fn test_operation(target_op: BinaryOp, name: &str) {
    let truth_table = TruthTable::for_operation(target_op);

    // Use reference implementation's learning rate range (0.05-0.06)
    let mut trainer = GateTrainer::new(0.05);

    let result = trainer.train(
        &truth_table,
        2000,      // max iterations
        1e-4,      // target loss (more realistic for convergence)
        true,      // verbose
    );

    println!("\n{} Training Result:", name);
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Final Loss: {:.8}", result.final_loss);
    println!("  Hard Accuracy: {:.2}%", result.hard_accuracy * 100.0);
    println!("  Dominant Op: {:?} ({:.2}%)", result.dominant_op, result.dominant_prob * 100.0);

    // Check exit criteria
    if result.meets_exit_criteria(target_op) {
        println!("  ✓ PASSED: Meets Phase 0.1 exit criteria");
    } else {
        println!("  ✗ FAILED: Does not meet exit criteria");
        if result.hard_accuracy <= 0.99 {
            println!("    - Hard accuracy too low: {:.2}%", result.hard_accuracy * 100.0);
        }
        if result.dominant_op != target_op {
            println!("    - Wrong dominant operation: {:?} instead of {:?}",
                result.dominant_op, target_op);
        }
        if !result.converged {
            println!("    - Did not converge");
        }
    }

    // Show probability distribution
    let probs = trainer.gate.probabilities();
    println!("\n  Final probability distribution:");
    for (i, &op) in BinaryOp::ALL.iter().enumerate() {
        if probs[i] > 0.01 {  // Only show operations with >1% probability
            println!("    {:?}: {:.4}", op, probs[i]);
        }
    }
}
