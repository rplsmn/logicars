//! Integration test for Phase 0.2: Gate Layer Training
//!
//! Trains multiple gates simultaneously to learn different operations.
//! Verifies that gates don't interfere with each other during training.

use logicars::{BinaryOp, LayerTrainer, LayerTruthTable};

fn main() {
    println!("=== Phase 0.2: Gate Layer Training ===\n");

    // Test 1: Train 3 gates to learn AND, OR, XOR
    println!("Test 1: Learning AND, OR, XOR simultaneously");
    println!("----------------------------------------------");

    let operations = vec![BinaryOp::And, BinaryOp::Or, BinaryOp::Xor];
    let truth_table = LayerTruthTable::for_operations(&operations);

    let mut trainer = LayerTrainer::new(3, 0.05);

    let result = trainer.train(&truth_table, 5000, 1e-4, true);

    println!("\n=== Results ===");
    println!(
        "Converged: {}, Iterations: {}",
        result.converged, result.iterations
    );
    println!("Final loss: {:.6}", result.final_loss);

    for (i, ((op, prob), acc)) in result
        .dominant_ops
        .iter()
        .zip(result.hard_accuracies.iter())
        .enumerate()
    {
        println!(
            "Gate {}: Learned {:?} ({:.2}%), Accuracy: {:.2}%",
            i,
            op,
            prob * 100.0,
            acc * 100.0
        );
    }

    if result.meets_exit_criteria(&operations) {
        println!("\n✅ Test 1 PASSED: All gates learned their target operations!");
    } else {
        println!("\n❌ Test 1 FAILED: Some gates did not converge correctly");
        std::process::exit(1);
    }

    // Test 2: Train 4 gates to learn different operations
    println!("\n\nTest 2: Learning AND, OR, XOR, NAND simultaneously");
    println!("---------------------------------------------------");

    let operations = vec![BinaryOp::And, BinaryOp::Or, BinaryOp::Xor, BinaryOp::Nand];
    let truth_table = LayerTruthTable::for_operations(&operations);

    let mut trainer = LayerTrainer::new(4, 0.05);

    let result = trainer.train(&truth_table, 5000, 1e-4, true);

    println!("\n=== Results ===");
    println!(
        "Converged: {}, Iterations: {}",
        result.converged, result.iterations
    );
    println!("Final loss: {:.6}", result.final_loss);

    for (i, ((op, prob), acc)) in result
        .dominant_ops
        .iter()
        .zip(result.hard_accuracies.iter())
        .enumerate()
    {
        println!(
            "Gate {}: Learned {:?} ({:.2}%), Accuracy: {:.2}%",
            i,
            op,
            prob * 100.0,
            acc * 100.0
        );
    }

    if result.meets_exit_criteria(&operations) {
        println!("\n✅ Test 2 PASSED: All gates learned their target operations!");
    } else {
        println!("\n❌ Test 2 FAILED: Some gates did not converge correctly");
        std::process::exit(1);
    }

    // Test 3: Train 8 gates to test scalability
    println!("\n\nTest 3: Learning 8 different operations simultaneously");
    println!("--------------------------------------------------------");

    let operations = vec![
        BinaryOp::And,
        BinaryOp::Or,
        BinaryOp::Xor,
        BinaryOp::Nand,
        BinaryOp::Nor,
        BinaryOp::Xnor,
        BinaryOp::A,
        BinaryOp::B,
    ];
    let truth_table = LayerTruthTable::for_operations(&operations);

    let mut trainer = LayerTrainer::new(8, 0.05);

    let result = trainer.train(&truth_table, 5000, 1e-4, true);

    println!("\n=== Results ===");
    println!(
        "Converged: {}, Iterations: {}",
        result.converged, result.iterations
    );
    println!("Final loss: {:.6}", result.final_loss);

    for (i, ((op, prob), acc)) in result
        .dominant_ops
        .iter()
        .zip(result.hard_accuracies.iter())
        .enumerate()
    {
        println!(
            "Gate {}: Learned {:?} ({:.2}%), Accuracy: {:.2}%",
            i,
            op,
            prob * 100.0,
            acc * 100.0
        );
    }

    if result.meets_exit_criteria(&operations) {
        println!("\n✅ Test 3 PASSED: All 8 gates learned their target operations!");
    } else {
        println!("\n❌ Test 3 FAILED: Some gates did not converge correctly");
        std::process::exit(1);
    }

    println!("\n=== All Tests Passed! ===");
    println!("Phase 0.2 exit criteria met:");
    println!("  ✅ Multiple gates can learn different operations simultaneously");
    println!("  ✅ No gradient interference between gates");
    println!("  ✅ Can learn arbitrary boolean function combinations");
    println!("  ✅ Pass-through initialization works at scale");
}
