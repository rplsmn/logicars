//! Test generalization of a trained checkerboard model on different grid sizes.
//!
//! Usage:
//!   cargo run --bin test_generalization --release -- <model.json>
//!
//! Tests the model on 16×16 (training size), 32×32, 64×64, and 128×128 grids.

use logicars::{
    compute_checkerboard_accuracy, create_checkerboard, create_random_seed, HardCircuit, SimpleRng,
    CHECKERBOARD_CHANNELS, CHECKERBOARD_SQUARE_SIZE,
};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse model path (required)
    let model_path = args.get(1).filter(|a| !a.starts_with("--"));
    if model_path.is_none() {
        eprintln!("Usage: test_generalization <model.json>");
        eprintln!("\nTest trained model on different grid sizes.");
        std::process::exit(1);
    }
    let model_path = model_path.unwrap();

    println!("=== Generalization Testing ===\n");
    println!("Loading model from: {}", model_path);

    // Load the HardCircuit
    let circuit = match HardCircuit::load(model_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    };

    println!(
        "Model loaded: {} channels, {} gates ({} active)",
        circuit.channels,
        circuit.total_gate_count(),
        circuit.active_gate_count()
    );

    // Test on different grid sizes
    let test_configs = [
        (16, 20, "training size"),
        (32, 40, "2× scale"),
        (64, 80, "4× scale - EXIT CRITERION"),
        (128, 160, "8× scale"),
    ];

    let mut rng = SimpleRng::new(42);
    let num_trials = 5;

    println!("\n=== Running Generalization Tests ===");
    println!(
        "{:>8} {:>8} {:>8} {:>12} {:>12}",
        "Size", "Steps", "Trials", "Accuracy", "Status"
    );
    println!("{:->8} {:->8} {:->8} {:->12} {:->12}", "", "", "", "", "");

    let mut all_passed = true;

    for (size, steps, desc) in test_configs {
        let target = create_checkerboard(size, CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_CHANNELS);

        // Run multiple trials and average
        let mut total_acc = 0.0;
        for _ in 0..num_trials {
            let seed = create_random_seed(size, CHECKERBOARD_CHANNELS, &mut rng);
            let output = circuit.run_steps(&seed, steps);
            let acc = compute_checkerboard_accuracy(&output, &target);
            total_acc += acc;
        }
        let avg_acc = total_acc / num_trials as f64;

        // Check exit criterion for 64×64
        let status = if size == 64 {
            if avg_acc >= 0.95 {
                "✅ PASS"
            } else {
                all_passed = false;
                "❌ FAIL"
            }
        } else {
            if avg_acc >= 0.90 {
                "✓"
            } else {
                "△"
            }
        };

        println!(
            "{:>8} {:>8} {:>8} {:>11.1}% {:>12}",
            format!("{}×{}", size, size),
            steps,
            num_trials,
            avg_acc * 100.0,
            status
        );

        // Print description
        if size == 64 {
            println!("         └─ {} (>95% required)", desc);
        }
    }

    println!("\n=== Summary ===");
    if all_passed {
        println!("✅ All exit criteria met!");
        println!("   64×64 generalization: PASSED");
    } else {
        println!("❌ Exit criteria NOT met");
        println!("   64×64 grid needs >95% accuracy");
    }

    println!("\n=== Done ===");
}
