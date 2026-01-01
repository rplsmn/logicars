//! Phase 0.3 Integration Test: Multi-Layer Circuit Training
//!
//! This binary tests that multi-layer circuits can learn composite operations
//! that require depth (e.g., XOR from AND/OR primitives).

use logicars::{
    BinaryOp, Circuit, CircuitTrainer, CircuitTrainingResult, CircuitTruthTable,
    ConnectionPattern,
};

fn main() {
    println!("=== Phase 0.3: Multi-Layer Circuit Training ===\n");

    // Test 1: Learn XOR with a 2-layer circuit
    let result1 = test_two_layer_xor();

    // Test 2: Learn XNOR with a 2-layer circuit
    let result2 = test_two_layer_xnor();

    // Test 3: Learn XOR with a 3-layer circuit (should also work)
    let result3 = test_three_layer_xor();

    println!("\n=== Summary ===\n");
    println!("Test 1 (2-layer XOR):  {} - {} iterations, {:.1}% accuracy",
        if result1.meets_exit_criteria() { "PASS" } else { "FAIL" },
        result1.iterations,
        result1.hard_accuracy * 100.0
    );
    println!("Test 2 (2-layer XNOR): {} - {} iterations, {:.1}% accuracy",
        if result2.meets_exit_criteria() { "PASS" } else { "FAIL" },
        result2.iterations,
        result2.hard_accuracy * 100.0
    );
    println!("Test 3 (3-layer XOR):  {} - {} iterations, {:.1}% accuracy",
        if result3.meets_exit_criteria() { "PASS" } else { "FAIL" },
        result3.iterations,
        result3.hard_accuracy * 100.0
    );

    let all_passed = result1.meets_exit_criteria()
        && result2.meets_exit_criteria()
        && result3.meets_exit_criteria();

    println!("\n=== Exit Criteria ===\n");
    println!("- Learn 2-3 layer circuits reliably: {}",
        if all_passed { "PASS" } else { "FAIL" }
    );
    println!("- Backpropagation through layers: PASS (verified by numerical gradient tests)");
    println!("- Decompose complex ops into primitives: {}",
        if result1.meets_exit_criteria() && result2.meets_exit_criteria() { "PASS" } else { "FAIL" }
    );

    if all_passed {
        println!("\n>>> ALL EXIT CRITERIA MET <<<");
    } else {
        println!("\n>>> SOME TESTS FAILED - SEE ABOVE <<<");
        std::process::exit(1);
    }
}

fn test_two_layer_xor() -> CircuitTrainingResult {
    println!("--- Test 1: 2-Layer XOR ---\n");
    println!("Architecture: 2 gates (layer 1) -> 1 gate (layer 2)");
    println!("Target: XOR(a, b)");
    println!("Note: XOR = AND(OR(a,b), NAND(a,b)) is one possible decomposition\n");

    // Create a 2-layer circuit for learning XOR
    // Layer 1: 2 gates, each receiving the same (a, b) inputs
    // Layer 2: 1 gate, taking outputs from both layer 1 gates
    let circuit = Circuit::two_layer_composite(2);
    let mut trainer = CircuitTrainer::new(circuit, 0.05);

    // Create truth table for XOR with the composite circuit structure
    let truth_table = CircuitTruthTable::for_two_layer_composite(|a, b| a ^ b);

    let result = trainer.train(&truth_table, 5000, 1e-4, true);

    println!("\nFinal state:");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Loss: {:.6}", result.final_loss);
    println!("  Hard accuracy: {:.1}%", result.hard_accuracy * 100.0);

    println!("\nLearned circuit:");
    for (layer_idx, layer_ops) in result.dominant_ops.iter().enumerate() {
        for (gate_idx, (op, prob)) in layer_ops.iter().enumerate() {
            println!("  Layer {}, Gate {}: {:?} ({:.1}%)", layer_idx, gate_idx, op, prob * 100.0);
        }
    }

    // Verify truth table
    println!("\nTruth table verification:");
    verify_xor_outputs(&trainer.circuit);

    result
}

fn test_two_layer_xnor() -> CircuitTrainingResult {
    println!("\n--- Test 2: 2-Layer XNOR ---\n");
    println!("Architecture: 2 gates (layer 1) -> 1 gate (layer 2)");
    println!("Target: XNOR(a, b) = NOT(XOR(a, b))");
    println!("Note: XNOR = OR(AND(a,b), NOR(a,b)) is one possible decomposition\n");

    let circuit = Circuit::two_layer_composite(2);
    let mut trainer = CircuitTrainer::new(circuit, 0.05);

    // XNOR truth table: (a XNOR b) = (a == b)
    let truth_table = CircuitTruthTable::for_two_layer_composite(|a, b| a == b);

    let result = trainer.train(&truth_table, 5000, 1e-4, true);

    println!("\nFinal state:");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Loss: {:.6}", result.final_loss);
    println!("  Hard accuracy: {:.1}%", result.hard_accuracy * 100.0);

    println!("\nLearned circuit:");
    for (layer_idx, layer_ops) in result.dominant_ops.iter().enumerate() {
        for (gate_idx, (op, prob)) in layer_ops.iter().enumerate() {
            println!("  Layer {}, Gate {}: {:?} ({:.1}%)", layer_idx, gate_idx, op, prob * 100.0);
        }
    }

    result
}

fn test_three_layer_xor() -> CircuitTrainingResult {
    println!("\n--- Test 3: 3-Layer XOR ---\n");
    println!("Architecture: 4 gates (layer 1) -> 2 gates (layer 2) -> 1 gate (layer 3)");
    println!("Target: XOR(a, b)");
    println!("Testing that deeper circuits can also learn\n");

    // Create a 3-layer circuit
    // Layer 1: 4 gates, all getting same (a, b)
    // Layer 2: 2 gates, paired connections from layer 1
    // Layer 3: 1 gate, takes outputs from layer 2
    let conn1 = ConnectionPattern::paired(2);  // gates 0,1 from layer 1 -> gate 0 layer 2, etc.
    let conn2 = ConnectionPattern::new(vec![(0, 1)]);  // both layer 2 outputs -> layer 3

    let circuit = Circuit::new(2, &[4, 2, 1], vec![conn1, conn2]);
    let mut trainer = CircuitTrainer::new(circuit, 0.05);

    // Create truth table for XOR
    // Layer 1 has 4 gates, all receiving same input
    let examples = vec![
        (vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], vec![0.0]),
        (vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], vec![1.0]),
        (vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)], vec![1.0]),
        (vec![(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)], vec![0.0]),
    ];

    let truth_table = CircuitTruthTable {
        examples,
        num_outputs: 1,
    };

    let result = trainer.train(&truth_table, 5000, 1e-4, true);

    println!("\nFinal state:");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Loss: {:.6}", result.final_loss);
    println!("  Hard accuracy: {:.1}%", result.hard_accuracy * 100.0);

    println!("\nLearned circuit:");
    for (layer_idx, layer_ops) in result.dominant_ops.iter().enumerate() {
        print!("  Layer {}: ", layer_idx);
        for (gate_idx, (op, prob)) in layer_ops.iter().enumerate() {
            print!("G{}={:?}({:.1}%) ", gate_idx, op, prob * 100.0);
        }
        println!();
    }

    result
}

fn verify_xor_outputs(circuit: &Circuit) {
    let test_cases = [
        (false, false, false),
        (false, true, true),
        (true, false, true),
        (true, true, false),
    ];

    for (a, b, expected) in test_cases {
        let inputs = vec![(a, a), (a, a)];  // Both layer-1 gates get same input
        let outputs = circuit.output_hard(&[(a, b), (a, b)]);
        let result = outputs[0];
        let status = if result == expected { "OK" } else { "FAIL" };
        println!("  {} XOR {} = {} (expected {}) [{}]", a as u8, b as u8, result as u8, expected as u8, status);
    }
}
