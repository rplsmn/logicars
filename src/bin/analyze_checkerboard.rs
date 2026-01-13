//! Analyze a trained checkerboard HardCircuit model.
//!
//! Usage:
//!   cargo run --bin analyze_checkerboard --release -- <model.json>
//!
//! Outputs:
//! - Gate distribution (count of each of 16 operations)
//! - Active vs pass-through ratio
//! - Optional CSV export with --csv=FILE

use logicars::{BinaryOp, Float, HardCircuit};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse model path (required)
    let model_path = args.get(1).filter(|a| !a.starts_with("--"));
    if model_path.is_none() {
        eprintln!("Usage: analyze_checkerboard <model.json> [--csv=FILE]");
        eprintln!("\nAnalyze a trained HardCircuit model.");
        std::process::exit(1);
    }
    let model_path = model_path.unwrap();

    // Parse --csv=FILE for CSV export
    let csv_path: Option<String> = args
        .iter()
        .find(|a| a.starts_with("--csv="))
        .and_then(|a| a.strip_prefix("--csv="))
        .map(|s| s.to_string());

    // Load model
    println!("=== Checkerboard Model Analysis ===\n");
    println!("Loading model from: {}", model_path);

    let circuit = match HardCircuit::load(model_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    };

    // Basic stats
    let total_gates = circuit.total_gate_count();
    let active_gates = circuit.active_gate_count();
    let pass_through = total_gates - active_gates;
    let active_pct = 100.0 * active_gates as Float / total_gates as Float;

    println!("\n=== Model Statistics ===");
    println!("  Channels: {}", circuit.channels);
    println!("  Perception gates: {}", circuit.perception.total_gate_count());
    println!("  Update gates: {}", circuit.update.total_gate_count());
    println!("  Total gates: {}", total_gates);
    println!("  Active gates: {} ({:.1}%)", active_gates, active_pct);
    println!("  Pass-through gates: {}", pass_through);

    // Gate distribution
    let dist = circuit.gate_distribution();
    println!("\n=== Gate Distribution ===");
    println!("{:>12} {:>8} {:>8}", "Operation", "Count", "Percent");
    println!("{:->12} {:->8} {:->8}", "", "", "");

    for (i, &count) in dist.iter().enumerate() {
        let op = BinaryOp::ALL[i];
        let pct = 100.0 * count as Float / total_gates as Float;
        let marker = if i == 3 || i == 5 { " (pass)" } else { "" };
        println!("{:>12?} {:>8} {:>7.1}%{}", op, count, pct, marker);
    }

    // Most common non-pass-through operations
    let mut ops_with_counts: Vec<(usize, usize)> = dist.iter().copied().enumerate().collect();
    ops_with_counts.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n=== Top Active Operations ===");
    for (i, count) in ops_with_counts.iter().take(5) {
        if *i != 3 && *i != 5 && *count > 0 {
            let op = BinaryOp::ALL[*i];
            println!("  {:?}: {} gates", op, count);
        }
    }

    // Perception vs Update breakdown
    println!("\n=== Perception Module ===");
    let perc_total = circuit.perception.total_gate_count();
    let perc_active = circuit.perception.active_gate_count();
    println!(
        "  {} kernels, {} gates ({} active, {:.1}%)",
        circuit.perception.kernels.len(),
        perc_total,
        perc_active,
        100.0 * perc_active as Float / perc_total as Float
    );

    println!("\n=== Update Module ===");
    let upd_total = circuit.update.total_gate_count();
    let upd_active = circuit.update.active_gate_count();
    println!(
        "  {} layers, {} gates ({} active, {:.1}%)",
        circuit.update.layers.len(),
        upd_total,
        upd_active,
        100.0 * upd_active as Float / upd_total as Float
    );

    // CSV export
    if let Some(ref csv) = csv_path {
        use std::fs::File;
        use std::io::Write;

        let mut f = File::create(csv).expect("Failed to create CSV file");
        
        // Overall distribution
        writeln!(f, "# Overall Gate Distribution").unwrap();
        writeln!(f, "operation,index,count,percent").unwrap();
        for (i, &count) in dist.iter().enumerate() {
            let op = BinaryOp::ALL[i];
            let pct = 100.0 * count as Float / total_gates as Float;
            writeln!(f, "{:?},{},{},{:.2}", op, i, count, pct).unwrap();
        }
        
        // Perception distribution
        writeln!(f, "").unwrap();
        writeln!(f, "# Perception Module Gate Distribution").unwrap();
        writeln!(f, "operation,index,count,percent").unwrap();
        let perc_dist = circuit.perception.gate_distribution();
        let perc_total = circuit.perception.total_gate_count();
        for (i, &count) in perc_dist.iter().enumerate() {
            let op = BinaryOp::ALL[i];
            let pct = 100.0 * count as Float / perc_total as Float;
            writeln!(f, "{:?},{},{},{:.2}", op, i, count, pct).unwrap();
        }
        
        // Update distribution
        writeln!(f, "").unwrap();
        writeln!(f, "# Update Module Gate Distribution").unwrap();
        writeln!(f, "operation,index,count,percent").unwrap();
        let upd_dist = circuit.update.gate_distribution();
        let upd_total = circuit.update.total_gate_count();
        for (i, &count) in upd_dist.iter().enumerate() {
            let op = BinaryOp::ALL[i];
            let pct = 100.0 * count as Float / upd_total as Float;
            writeln!(f, "{:?},{},{},{:.2}", op, i, count, pct).unwrap();
        }
        
        println!("\n=== Export ===");
        println!("  Gate distribution saved to: {}", csv);
    }

    println!("\n=== Done ===");
}
